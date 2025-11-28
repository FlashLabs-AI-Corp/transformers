# Copyright 2025 The FlashLabs team. All rights reserved.
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

import json
import shutil
import tempfile
import unittest

import numpy as np

from transformers import ChromaProcessor
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


@require_torch
class ChromaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ChromaProcessor
    audio_input_name = "input_values"

    @classmethod
    def setUpClass(cls):
        # TODO: update with correct Chroma model checkpoint
        cls.checkpoint = "/models/Qwencsm/Chroma/checkpoints/chroma_transformers_1119"
        # For now, we'll skip the actual loading and just test the structure
        cls.tmpdirname = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def prepare_processor_dict(self):
        # Chroma uses Qwen2.5Omni's chat template structure
        return {
            "chat_template": "{% for message in messages %}{{ message['role'] }}: {{ message['content'][0]['text'] }}{% endfor %}"
        }

    @unittest.skip(reason="Chroma processor requires specific checkpoint structure")
    def test_chat_template_is_saved(self):
        pass

    @require_torch
    @unittest.skip(reason="Chroma has custom apply_chat_template implementation")
    def test_apply_chat_template(self):
        pass

    @require_torch
    @unittest.skip(reason="Chroma doesn't use assistant masks as an audio generation model")
    def test_apply_chat_template_assistant_mask(self):
        pass

    def test_processor_structure(self):
        """Test that ChromaProcessor has the expected structure"""
        # This is a basic structural test
        processor_attributes = dir(ChromaProcessor)
        
        # Check for essential methods
        self.assertIn("__call__", processor_attributes)
        self.assertIn("apply_chat_template", processor_attributes)
        self.assertIn("load_audio", processor_attributes)

    @require_torch
    def test_load_audio_basic(self):
        """Test basic audio loading functionality"""
        processor = ChromaProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B-Instruct")
        
        # Create a dummy audio file
        import tempfile
        import torchaudio
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create a simple sine wave
            sample_rate = 16000
            duration = 1  # 1 second
            frequency = 440  # A4 note
            t = torch.linspace(0, duration, int(sample_rate * duration))
            waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
            
            torchaudio.save(f.name, waveform, sample_rate)
            audio_path = f.name
        
        try:
            # Test loading audio
            audio_tensor = processor.load_audio(audio_path, target_sample_rate=24000)
            
            # Check output shape and type
            self.assertIsInstance(audio_tensor, torch.Tensor)
            self.assertEqual(audio_tensor.dim(), 1)  # Should be 1D tensor
            
            # Check that resampling worked (approximately)
            expected_length = int(24000 * duration)
            self.assertAlmostEqual(len(audio_tensor), expected_length, delta=100)
        finally:
            import os
            os.unlink(audio_path)

    @require_torch
    def test_processor_call_structure(self):
        """Test that processor __call__ returns expected structure"""
        processor = ChromaProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B-Instruct")
        
        # Create dummy inputs
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Test message"},
                        {"type": "audio", "audio": np.random.randn(16000)},
                    ],
                }
            ]
        ]
        
        prompt_text = ["Test prompt"]
        
        # Create a dummy audio file for prompt_audio
        import tempfile
        import torchaudio
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sample_rate = 24000
            duration = 1
            t = torch.linspace(0, duration, int(sample_rate * duration))
            waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
            torchaudio.save(f.name, waveform, sample_rate)
            audio_path = f.name
        
        try:
            prompt_audio = [audio_path]
            
            # Call processor
            inputs = processor(
                conversations=conversations,
                prompt_text=prompt_text,
                prompt_audio=prompt_audio,
                return_tensors="pt"
            )
            
            # Check that expected keys are present
            self.assertIn("input_ids", inputs)
            self.assertIn("attention_mask", inputs)
            self.assertIn("input_values", inputs)
            self.assertIn("input_values_cutoffs", inputs)
            
            # Check that thinker inputs are present
            self.assertIn("thinker_input_ids", inputs)
            self.assertIn("thinker_attention_mask", inputs)
            
            # Check tensor types
            self.assertIsInstance(inputs["input_ids"], torch.Tensor)
            self.assertIsInstance(inputs["attention_mask"], torch.Tensor)
            self.assertIsInstance(inputs["input_values"], torch.Tensor)
            self.assertIsInstance(inputs["input_values_cutoffs"], torch.Tensor)
            
            # Check batch size
            batch_size = len(conversations)
            self.assertEqual(inputs["input_ids"].shape[0], batch_size)
            self.assertEqual(inputs["input_values"].shape[0], batch_size)
            
        finally:
            import os
            os.unlink(audio_path)

    @require_torch
    def test_processor_batched_inputs(self):
        """Test processor with batched inputs"""
        processor = ChromaProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B-Instruct")
        
        # Create dummy inputs with batch size 2
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First message"},
                        {"type": "audio", "audio": np.random.randn(16000)},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Second message"},
                        {"type": "audio", "audio": np.random.randn(16000)},
                    ],
                }
            ],
        ]
        
        prompt_text = ["Test prompt 1", "Test prompt 2"]
        
        # Create dummy audio files
        import tempfile
        import torchaudio
        
        audio_paths = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sample_rate = 24000
                duration = 1
                t = torch.linspace(0, duration, int(sample_rate * duration))
                waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
                torchaudio.save(f.name, waveform, sample_rate)
                audio_paths.append(f.name)
        
        try:
            prompt_audio = audio_paths
            
            # Call processor
            inputs = processor(
                conversations=conversations,
                prompt_text=prompt_text,
                prompt_audio=prompt_audio,
                return_tensors="pt"
            )
            
            # Check batch size
            batch_size = 2
            self.assertEqual(inputs["input_ids"].shape[0], batch_size)
            self.assertEqual(inputs["input_values"].shape[0], batch_size)
            self.assertEqual(inputs["thinker_input_ids"].shape[0], batch_size)
            
            # Check that input_values_cutoffs has correct length
            self.assertEqual(len(inputs["input_values_cutoffs"]), batch_size)
            
        finally:
            import os
            for path in audio_paths:
                os.unlink(path)

    @require_torch
    def test_apply_chat_template_basic(self):
        """Test apply_chat_template method"""
        processor = ChromaProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B-Instruct")
        
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "audio", "audio": np.random.randn(16000)},
                ],
            }
        ]
        
        # Test that apply_chat_template returns text and audios
        text, audios = processor.apply_chat_template(conversations, tokenize=False)
        
        self.assertIsInstance(text, list)
        self.assertEqual(len(text), 1)
        self.assertIsInstance(text[0], str)
        
        if audios is not None:
            self.assertIsInstance(audios, list)

    def test_processor_inheritance(self):
        """Test that ChromaProcessor inherits from Qwen2_5OmniProcessor"""
        from transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor
        
        self.assertTrue(issubclass(ChromaProcessor, Qwen2_5OmniProcessor))

    @require_torch
    def test_audio_resampling(self):
        """Test that audio resampling works correctly"""
        processor = ChromaProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B-Instruct")
        
        # Create audio at 16kHz
        import tempfile
        import torchaudio
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sample_rate = 16000
            duration = 2  # 2 seconds
            t = torch.linspace(0, duration, int(sample_rate * duration))
            waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
            torchaudio.save(f.name, waveform, sample_rate)
            audio_path = f.name
        
        try:
            # Load and resample to 24kHz
            audio_24k = processor.load_audio(audio_path, target_sample_rate=24000)
            expected_length_24k = int(24000 * duration)
            self.assertAlmostEqual(len(audio_24k), expected_length_24k, delta=100)
            
            # Load and resample to 8kHz
            audio_8k = processor.load_audio(audio_path, target_sample_rate=8000)
            expected_length_8k = int(8000 * duration)
            self.assertAlmostEqual(len(audio_8k), expected_length_8k, delta=100)
            
        finally:
            import os
            os.unlink(audio_path)

