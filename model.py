# Copyright (2023) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    WhisperProcessor,
    LlamaForCausalLM,
    LlamaTokenizer,
    WhisperForConditionalGeneration
)
import librosa
from beats.BEATs import BEATsConfig, BEATs
from qformer.Qformer import BertConfig, BertLMHeadModel

import torch.optim as optim
import random
from tqdm import tqdm
from CustomWhisper import WhisperFeatureExtractorTorch
import numpy as np
import csv


class SALMONN(nn.Module):
    def __init__(
        self,
        ckpt,
        whisper_path,
        beats_path,
        vicuna_path,
        device="cuda:0",
        speech_qformer_token_num=1,
        speech_qformer_layer=2,
        lora=True,
        lora_alpha=32,
        lora_rank=8,
        lora_dropout=0.1,
        second_per_frame=0.333333,
        second_stride=0.333333,
        low_resource=False
    ):

        super().__init__()

        # feature_extractor
        self.feature_extractor = WhisperFeatureExtractorTorch(feature_size=80, sampling_rate=16000, hop_length=160, n_fft=400, device=device)
        self.feature_extractor_notorch = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_path, device=device)
        self.whisper_generator = WhisperForConditionalGeneration.from_pretrained(whisper_path)

        # whisper
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)

        # beats
        self.beats_ckpt = beats_path
        beats_checkpoint = torch.load(self.beats_ckpt, map_location='cpu')
        beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
        beats = BEATs(beats_cfg)
        beats.load_state_dict(beats_checkpoint['model'])
        self.beats = beats
        self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
        for name, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()

        # init speech Qformer
        self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            speech_qformer_token_num,
            self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim,
            speech_qformer_layer,
        )
        self.second_per_frame = second_per_frame
        self.second_stride = second_stride
        
        # vicuna
        cuda_num = int(device.split(":")[-1])
        if not low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=torch.float16,
                device_map={'': cuda_num}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': cuda_num}
            )

        # lora
        self.lora = lora
        if lora:
            target_modules = None
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=True, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)

        # tokenizer
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        self.llama_tokenizer.padding_side = "right"

        # proj
        self.speech_llama_proj = nn.Linear(
            self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size)

        # load ckpt
        ckpt_dict = torch.load(ckpt, map_location=device)['model']
        self.load_state_dict(ckpt_dict, strict=False)

        # if combined_ckpt_path:
        #     print(f"Loading combined checkpoint from {combined_ckpt_path}")
        #     self.load_state_dict(torch.load(combined_ckpt_path))
        # elif ckpt:
        #     ckpt_dict = torch.load(ckpt)['model']
        #     self.load_state_dict(ckpt_dict, strict=False)

    def generate(
        self,
        wav_path,
        prompt,
        prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        device='cuda:0',
        max_length=300,
        num_beams=4,
        do_sample=True,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
    ):

        # read wav
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
        
        # whisper
        #spectrogram = self.feature_extractor.extract_fbank_features(wav)
        spectrogram = self.feature_extractor_notorch(wav, return_tensors="pt", sampling_rate=16000).input_features.to(device) # [1, 80, 3000]
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
       
        # beats
        raw_wav = torch.from_numpy(wav).to(device).unsqueeze(0)
        audio_padding_mask = torch.zeros(raw_wav.shape, device=device).bool()
        audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)

        # auditory embeds
        speech_embeds = self.ln_speech(speech_embeds)
        audio_embeds = self.ln_audio(audio_embeds)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

        # split frames
        B, T, C = speech_embeds.shape
        kernel = round(T * self.second_per_frame / 30.0)
        stride = round(T * self.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C).to(device)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        # Qformer
        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        prompt_left, prompts_right = prompt_pattern.format(prompt).split('<SpeechHere>')
        prompt_left_ids = self.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)
        prompt_right_ids = self.llama_tokenizer(
            prompts_right,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids)

        bos_embeds = self.llama_model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=device,
            ) * self.llama_tokenizer.bos_token_id
        ) if not self.lora else self.llama_model.model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=device,
            ) * self.llama_tokenizer.bos_token_id
        )

        embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        
        # generate
        output = self.llama_model.generate(
            inputs_embeds=embeds,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            attention_mask=atts,
            bos_token_id=self.llama_tokenizer.bos_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            output_scores = True
            #return_dict_in_generate=True
        )
        
        output_text = self.llama_tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=True)

        return output_text

    def init_speech_Qformer(self, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens


    def generate_jailbreak(
        self,
        wav_path, # path to audio file to optimize
        name, # name of experiment for saving
        target_text, # text to optimize for
        prompt, # prompt accompanying optimization
        prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        device='cuda:0',
        batch_size=8,
        max_length=100,
        lr=0.01,
        epsilon=None,
        freq_clipping=None,
        num_iterations=200,
        lr_step = 1001,
        optimization_method = "gd",
        logging = False):


        # Read wav
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")

        if logging:
            beats_checkpoint = torch.load(self.beats_ckpt, map_location='cpu')
            log_file = f'training_logs/{name}.csv'
            log_columns = ['step', 'total_loss', 'learning_rate', 'beats_features', 'whisper_features']
    
            with open(log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(log_columns)
            
        wav_tensor = torch.tensor(wav, device=device, requires_grad=True)
        orig_wav_tensor = wav_tensor.detach().clone()
        
        if optimization_method == "gd":
            optimizer = torch.optim.AdamW([wav_tensor], lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

        wav_fft_orig = torch.fft.rfft(orig_wav_tensor)  # FFT of original audio
        #orig_amplitudes = torch.abs(wav_fft_orig)  # Magnitude of original frequencies

        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        
        for t in tqdm(range(num_iterations + 1)):
            if optimization_method == "gd":
                optimizer.zero_grad()

            sampled_target_text = random.sample(target_text, batch_size)
            to_regress_tokens = self.llama_tokenizer(
                sampled_target_text,
                add_special_tokens=False,
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_tensors="pt" 
            ).to(device)
            target_ids = to_regress_tokens.input_ids
            target_ids = target_ids.clamp(max=embed_tokens.weight.shape[0] - 1) #rly weird but necessary apparently
            embedded_targets = embed_tokens(target_ids)

            bos = torch.ones([1, 1],
                         dtype=target_ids.dtype,
                         device=target_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embs = embed_tokens(bos)

            pad = torch.ones([1, 1],
                         dtype=target_ids.dtype,
                         device=target_ids.device) * (self.llama_tokenizer.pad_token_id - 1)
            pad_embs = embed_tokens(pad)
            
            T = target_ids.clone()  # Cloning to prevent unintended in-place modification
            T = T.masked_fill(T == self.llama_tokenizer.pad_token_id - 1, -100)
            pos_padding = torch.argmin(T, dim=1)

            # Custom Whisper processing (audio to embeddings)
            spectrogram = self.feature_extractor.extract_fbank_features(wav_tensor)
            speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

            if logging:
                input_features = self.whisper_processor.feature_extractor(wav_tensor.detach().cpu().numpy(), return_tensors="pt", sampling_rate=16000).input_features.to(device)
                generated_ids = self.whisper_generator.generate(input_features)
                transcription = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            """wav = wav_tensor.detach().cpu().numpy()
            spectrogram_orig = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(device) # [1, 80, 3000]"""
            
            raw_wav = wav_tensor.unsqueeze(0)
            audio_padding_mask = torch.zeros(raw_wav.shape, device=device).bool()
            audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)

            if logging:
                probs = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask)[0]
                for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
                    top5_label = [beats_checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
    
            # Auditory embeds
            speech_embeds = self.ln_speech(speech_embeds)
            audio_embeds = self.ln_audio(audio_embeds)
            audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
            speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

            # Prepare prompt embeddings
            prompt_left, prompts_right = prompt_pattern.format(prompt).split('<SpeechHere>')
            prompt_left_ids = self.llama_tokenizer(prompt_left, return_tensors="pt", add_special_tokens=False).to(device).input_ids
            prompt_left_embeds = embed_tokens(prompt_left_ids).repeat(batch_size, 1, 1)
            prompt_right_ids = self.llama_tokenizer(prompts_right, return_tensors="pt", add_special_tokens=False).to(device).input_ids
            prompt_right_embeds = embed_tokens(prompt_right_ids).repeat(batch_size, 1, 1)

            # Frame splitting for temporal alignment
            B, T2, C = speech_embeds.shape
            kernel = round(T2 * self.second_per_frame / 30.0)
            stride = round(T2 * self.second_stride / 30.0)
            kernel = (1, kernel)
            stride = (1, stride)
            speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
            speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
            _, _, L = speech_embeds_overlap.shape
            speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
            speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
            speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)
        
            # QFormer processing
            query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
            query_output = self.speech_Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=speech_embeds,
                encoder_attention_mask=speech_atts,
                return_dict=True,
            )
            
            speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
            speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous().repeat(batch_size, 1, 1)
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

            context_embs = torch.cat((prompt_left_embeds, speech_embeds, prompt_right_embeds), dim=1)

            input_embs = []
            targets_mask = []
    
            target_tokens_length = []
            context_tokens_length = []
            seq_tokens_length = []

            for i in range(batch_size):
                pos = int(pos_padding[i])
                if T[i][pos] == -100:
                    target_length = pos
                else:
                    target_length = T.shape[1]

                targets_mask.append(T[i:i+1, :target_length])
                input_embs.append(embedded_targets[i:i+1, :target_length]) # omit the padding tokens
    
                context_length = context_embs[i].unsqueeze(0).shape[1] #HERE
                seq_length = target_length + context_length
    
                target_tokens_length.append(target_length)
                context_tokens_length.append(context_length)
                seq_tokens_length.append(seq_length)

            max_seq_length = max(seq_tokens_length)

            attention_mask = []

            for i in range(batch_size):

                # masked out the context from loss computation
                context_mask =(
                    torch.ones([1, context_tokens_length[i] + 1],
                           dtype=torch.long).to(device).fill_(-100)  # plus one for bos
                )
    
                # padding to align the length
                num_to_pad = max_seq_length - seq_tokens_length[i]
                padding_mask = (
                    torch.ones([1, num_to_pad],
                           dtype=torch.long).to(device).fill_(-100)
                )
    
                targets_mask[i] = torch.cat( [context_mask, targets_mask[i], padding_mask], dim=1 )
                input_embs[i] = torch.cat( [bos_embs, context_embs[i].unsqueeze(0), input_embs[i],
                                            pad_embs.repeat(1, num_to_pad, 1)], dim=1 )
                
                attention_mask.append(torch.LongTensor( [[1]* (1+seq_tokens_length[i]) + [0]*num_to_pad]))
    
            targets = torch.cat( targets_mask, dim=0 ).to(device)
            inputs_embs = torch.cat( input_embs, dim=0 ).to(device)
            attention_mask = torch.cat(attention_mask, dim=0).to(device)

            outputs = self.llama_model(
                inputs_embeds=inputs_embs,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            likelihood_loss = outputs.loss
            
            """# Compute the difference in amplitudes between the current and original audio
            current_amplitudes = torch.abs(wav_fft) 
            amplitude_difference = current_amplitudes - orig_amplitudes
            amplitude_difference = torch.abs(current_amplitudes - orig_amplitudes)
            
            # Penalize only the added noise in the audible range
            audible_amplitude_penalty = (amplitude_difference * audible_mask).sum()
            frequency_loss = freq_constraint_weight * audible_amplitude_penalty
            total_loss = likelihood_loss + frequency_loss"""
            
            likelihood_loss.backward()

            if logging:
                with open(log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([t, likelihood_loss.item(), lr, top5_label, transcription])

            if t > 0 and t % lr_step == 0:
                print(f"Loss at step {t}: {likelihood_loss.item()}")
                lr = lr/10
                if optimization_method == "fsgm":
                    print(f"LR at step {t}: {lr}")
                else:
                    print(f"LR at step {t}: {optimizer.param_groups[0]['lr']}")
            
            if optimization_method == "fsgm":
                adjusted_wav = (wav_tensor.data - lr * wav_tensor.grad.detach().sign()).clamp(-1, 1)
                if epsilon:
                    wav_tensor.data = adjusted_wav.clamp(orig_wav_tensor - epsilon, orig_wav_tensor + epsilon)
            else:
                optimizer.step()
                scheduler.step()
                if epsilon:
                    wav_tensor.data = wav_tensor.data.clamp(orig_wav_tensor - epsilon, orig_wav_tensor + epsilon)

            wav_tensor.grad.zero_()

            if freq_clipping:
                wav_fft = torch.fft.rfft(wav_tensor)
                freqs = torch.fft.rfftfreq(wav_tensor.size(0), 1 / 16000) 
                audible_mask = ((freqs >= freq_clipping[0]) & (freqs <= freq_clipping[1])).float().to(device)
                wav_fft_preserved = wav_fft * (1 - audible_mask) + wav_fft_orig * audible_mask
                wav_cleaned = torch.fft.irfft(wav_fft_preserved, n=wav_tensor.size(0))
                wav_tensor.data = wav_cleaned

        return wav_tensor.detach().cpu().numpy()

    def optimize_prepend_audio(
        self,
        wav_path,
        target_text,  # text from derogatory_corpus
        prompt,
        prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        device='cuda:0',
        batch_size=8,
        max_length=150,
        epsilon=None,
        lr=0.01,
        num_iterations=500,
        prepend_duration=2,  # Duration of the prepended audio in seconds
        lr_step = 1001,
        fgsm=False
    ):
        # Read original wav
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
        
        wav_tensor = torch.tensor(wav, device=device, requires_grad=False)
        
        # Initialize new audio segment for prepending (random initialization)
        prepend_length = int(prepend_duration * 16000)  # 2 seconds at 16kHz
        prepend_tensor = torch.randn(prepend_length, device=device, requires_grad=True)
        
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        
        if not fgsm:
            optimizer = torch.optim.AdamW([prepend_tensor], lr=lr)
        
        for t in tqdm(range(num_iterations + 1)):
            if not fgsm:
                optimizer.zero_grad()

            with_prepend_wav_tensor = torch.cat([prepend_tensor, wav_tensor], dim=0)
            prepend_array = prepend_tensor.detach().cpu().numpy()
            with_prepend_wav = np.concatenate([prepend_array, wav])

            sampled_target_text = random.sample(target_text, batch_size)
            to_regress_tokens = self.llama_tokenizer(
                sampled_target_text,
                add_special_tokens=False,
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_tensors="pt" 
            ).to(device)
            target_ids = to_regress_tokens.input_ids
            target_ids = target_ids.clamp(max=embed_tokens.weight.shape[0] - 1) #rly weird but necessary apparently
            embedded_targets = embed_tokens(target_ids)

            bos = torch.ones([1, 1],
                         dtype=target_ids.dtype,
                         device=target_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embs = embed_tokens(bos)

            pad = torch.ones([1, 1],
                         dtype=target_ids.dtype,
                         device=target_ids.device) * (self.llama_tokenizer.pad_token_id - 1)
            pad_embs = embed_tokens(pad)
            
            T = target_ids.clone()  # Cloning to prevent unintended in-place modification
            T = T.masked_fill(T == self.llama_tokenizer.pad_token_id - 1, -100)
            pos_padding = torch.argmin(T, dim=1)

        
            # Custom Whisper processing (audio to embeddings)
            spectrogram = self.feature_extractor.extract_fbank_features(with_prepend_wav_tensor)

            """wav = wav_tensor.detach().cpu().numpy()
            spectrogram_orig = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(device) # [1, 80, 3000]"""
            
            speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
    
            # Beats
            raw_wav = torch.from_numpy(with_prepend_wav).to(device).unsqueeze(0)
            audio_padding_mask = torch.zeros(raw_wav.shape, device=device).bool()
            audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
    
            # Auditory embeds
            speech_embeds = self.ln_speech(speech_embeds)
            audio_embeds = self.ln_audio(audio_embeds)
            audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
            speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

            # Prepare prompt embeddings
            prompt_left, prompts_right = prompt_pattern.format(prompt).split('<SpeechHere>')
            prompt_left_ids = self.llama_tokenizer(prompt_left, return_tensors="pt", add_special_tokens=False).to(device).input_ids
            prompt_left_embeds = embed_tokens(prompt_left_ids).repeat(batch_size, 1, 1)
            prompt_right_ids = self.llama_tokenizer(prompts_right, return_tensors="pt", add_special_tokens=False).to(device).input_ids
            prompt_right_embeds = embed_tokens(prompt_right_ids).repeat(batch_size, 1, 1)

            # Frame splitting for temporal alignment
            B, T2, C = speech_embeds.shape
            kernel = round(T2 * self.second_per_frame / 30.0)
            stride = round(T2 * self.second_stride / 30.0)
            kernel = (1, kernel)
            stride = (1, stride)
            speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
            speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
            _, _, L = speech_embeds_overlap.shape
            speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
            speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
            speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)
        
            # QFormer processing
            query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
            query_output = self.speech_Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=speech_embeds,
                encoder_attention_mask=speech_atts,
                return_dict=True,
            )
            
            speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
            speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous().repeat(batch_size, 1, 1)
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

            context_embs = torch.cat((prompt_left_embeds, speech_embeds, prompt_right_embeds), dim=1)

            input_embs = []
            targets_mask = []
    
            target_tokens_length = []
            context_tokens_length = []
            seq_tokens_length = []

            for i in range(batch_size):
                pos = int(pos_padding[i])
                if T[i][pos] == -100:
                    target_length = pos
                else:
                    target_length = T.shape[1]

                targets_mask.append(T[i:i+1, :target_length])
                input_embs.append(embedded_targets[i:i+1, :target_length]) # omit the padding tokens
    
                context_length = context_embs[i].unsqueeze(0).shape[1] #HERE
                seq_length = target_length + context_length
    
                target_tokens_length.append(target_length)
                context_tokens_length.append(context_length)
                seq_tokens_length.append(seq_length)

            max_seq_length = max(seq_tokens_length)

            attention_mask = []

            for i in range(batch_size):

                # masked out the context from loss computation
                context_mask =(
                    torch.ones([1, context_tokens_length[i] + 1],
                           dtype=torch.long).to(device).fill_(-100)  # plus one for bos
                )
    
                # padding to align the length
                num_to_pad = max_seq_length - seq_tokens_length[i]
                padding_mask = (
                    torch.ones([1, num_to_pad],
                           dtype=torch.long).to(device).fill_(-100)
                )
    
                targets_mask[i] = torch.cat( [context_mask, targets_mask[i], padding_mask], dim=1 )
                input_embs[i] = torch.cat( [bos_embs, context_embs[i].unsqueeze(0), input_embs[i],
                                            pad_embs.repeat(1, num_to_pad, 1)], dim=1 )
                
                attention_mask.append(torch.LongTensor( [[1]* (1+seq_tokens_length[i]) + [0]*num_to_pad]))
    
            targets = torch.cat( targets_mask, dim=0 ).to(device)
            inputs_embs = torch.cat( input_embs, dim=0 ).to(device)
            attention_mask = torch.cat(attention_mask, dim=0).to(device)

            outputs = self.llama_model(
                inputs_embeds=inputs_embs,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            loss.backward()

            if fgsm:
                adjusted_wav = (prepend_tensor.data - lr * prepend_tensor.grad.detach().sign()).clamp(-1, 1)
                prepend_tensor.data = adjusted_wav
                if epsilon:
                    wav_tensor.data = adjusted_wav.clamp(orig_wav_tensor - epsilon, orig_wav_tensor + epsilon)
                
            else:
                optimizer.step()
                if epsilon:
                    wav_tensor.data = wav_tensor.clamp(orig_wav_tensor - epsilon, orig_wav_tensor + epsilon)
                
            prepend_tensor.grad.zero_()

            if t > 0 and t % lr_step == 0:
                print(f"Total Loss at step {t}: {loss.item()}")
                if not fgsm:
                    optimizer.param_groups[0]['lr'] /= 10
                    print(f"LR at step {t}: {optimizer.param_groups[0]['lr']}")
                else:
                    lr = lr/10
                    print(f"LR at step {t}: {lr}")


        prepend_snippet = prepend_tensor.detach().cpu().numpy()
        combined_audio = np.concatenate([prepend_snippet, wav])  # Combine the prepended snippet with the original audio
        return combined_audio


