# coding=utf-8
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
"""Testing suite for the PyTorch Chroma model."""

import copy
import unittest

import pytest
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    ChromaConfig,
    ChromaForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
)

if is_datasets_available():
    from datasets import load_dataset

if is_torch_available():
    import torch


class ChromaModelTester:
    def __init__(
            self,
            parent,
            ignore_index=-100,
            batch_size=2,
            seq_length=7,
            is_training=True,
            thinker_config={
                "model_type": "qwen2_5_omni_thinker",
                "audio_token_index": 151646,
                "image_token_index": 151655,
                "video_token_index": 151656,
                "bos_token_id": 151644,
                "eos_token_id": 151645,
                "pad_token_id": 151643,
                "audio_config": {
                    "model_type": "qwen2_5_omni_audio_encoder",
                    "d_model": 64,
                    "encoder_attention_heads": 2,
                    "encoder_ffn_dim": 128,
                    "encoder_layers": 2,
                    "num_mel_bins": 128,
                },
                "text_config": {
                    "model_type": "qwen2_5_omni_text",
                    "vocab_size": 151936,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "hidden_act": "silu",
                    "max_position_embeddings": 128,
                },
            },
            backbone_config={
                "model_type": "chroma_backbone",
                "audio_num_codebooks": 8,
                "vocab_size": 2051,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_act": "silu",
                "max_position_embeddings": 128,
            },
            decoder_config={
                "model_type": "chroma_decoder",
                "audio_num_codebooks": 8,
                "audio_embedding_dim": 64,
                "vocab_size": 2051,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_act": "silu",
                "max_position_embeddings": 33,
            },
            codec_config={
                "model_type": "mimi",
                "audio_channels": 1,
                "chunk_in_sec": None,
                "hidden_size": 32,
                "num_filters": 8,
                "num_residual_layers": 1,
                "upsampling_ratios": [8, 4],
                "codebook_size": 2048,
                "vector_quantization_hidden_dimension": 64,
                "upsample_groups": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "sliding_window": 4,
                "codebook_dim": 64,
                "use_cache": False,
                "num_quantizers": 8,
                "frame_rate": 12.5,
            },
            config={
                "audio_num_codebooks": 8,
                "codebook_pad_token_id": 2050,
                "codebook_eos_token_id": 0,
                "text_start_token_id": 151665,
                "text_end_token_id": 151666,
                "im_end_token_id": 151645,
                "audio_frame_freq": 1920,
            },
    ):
        self.parent = parent
        self.is_training = is_training
        self.ignore_index = ignore_index
        self.thinker_config = thinker_config
        self.backbone_config = backbone_config
        self.decoder_config = decoder_config
        self.codec_config = codec_config
        self.config = config
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.num_hidden_layers = backbone_config["num_hidden_layers"]
        self.vocab_size = backbone_config["vocab_size"]
        self.hidden_size = backbone_config["hidden_size"]
        self.num_attention_heads = backbone_config["num_attention_heads"]
        self.audio_num_codebooks = config["audio_num_codebooks"]

    def get_config(self):
        return ChromaConfig(
            thinker_config=self.thinker_config,
            backbone_config=self.backbone_config,
            decoder_config=self.decoder_config,
            codec_config=self.codec_config,
            **self.config,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length, config.audio_num_codebooks],
                               config.backbone_config.vocab_size - 1) + 1
        attention_mask = torch.ones([self.batch_size, self.seq_length], dtype=torch.long, device=torch_device)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


class ChromaForConditionalGenerationTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (ChromaForConditionalGeneration,) if is_torch_available() else ()

    test_resize_embeddings = False
    test_resize_embeddings_untied = False
    test_head_masking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = ChromaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ChromaConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        """
        Overrides [ModelTesterMixin._prepare_for_class] to handle third input_ids dimension.
        """
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (
                    self.model_tester.batch_size,
                    self.model_tester.seq_length,
                    self.model_tester.config["audio_num_codebooks"],
                ),
                dtype=torch.long,
                device=torch_device,
            )

        return inputs_dict

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        """
        Overrides [GenerationTesterMixin._get_logits_processor_kwargs] to restrict to top_k, top_p, and temperature sampling.
        """
        logits_processor_kwargs = {}
        if do_sample:
            logits_processor_kwargs.update(
                {
                    "top_k": 10,
                    "top_p": 0.7,
                    "temperature": 0.7,
                }
            )

        return logits_processor_kwargs

    def _check_similar_generate_outputs(self, output_1, output_2, atol=1e-5, rtol=1e-5):
        """
        Overrides [GenerationTesterMixin._check_similar_generate_outputs] to handle third input_ids dimension.
        Here we only look at the first codebook (index 0 on last dimension of the generated sequences) since returned scores
        are for this token.
        """
        # scores doesn't include data regarding decoder input tokens
        decoder_input_length = output_1.sequences.shape[1] - len(output_1.scores)
        output_matches = output_1.sequences[..., 0] == output_2.sequences[..., 0]
        has_matching_outputs = output_matches.all()
        has_matching_scores = None
        if not has_matching_outputs:
            for batch_idx in range(output_1.sequences.shape[0]):
                batch_matches = output_matches[batch_idx]
                if batch_matches.all():
                    continue
                first_mismatch_idx = batch_matches.int().argmin()  # gets the index of the first False
                first_mismatch_idx -= decoder_input_length
                output_1_first_mismatch_scores = output_1.scores[first_mismatch_idx][batch_idx]
                output_2_first_mismatch_scores = output_2.scores[first_mismatch_idx][batch_idx]
                has_matching_scores = torch.allclose(
                    output_1_first_mismatch_scores, output_2_first_mismatch_scores, rtol=atol, atol=rtol
                )
                if not has_matching_scores:
                    break
        self.assertTrue(has_matching_outputs or has_matching_scores)

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support assisted decoding.")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support beam search.")
    def test_beam_sample_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support beam search.")
    def test_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support beam search.")
    def test_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support beam search.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support beam search.")
    def test_beam_sample_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support prompt lookup decoding.")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support prompt lookup decoding.")
    def test_prompt_lookup_decoding_stops_at_eos(self):
        pass

    @pytest.mark.skip(reason="Chroma has custom embedding approach (text and audio embeddings).")
    def test_model_get_set_embeddings(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support beam search.")
    def test_generate_from_inputs_embeds_1_beam_search(self, _, num_beams):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Chroma does not support beam search.")
    def test_model_parallel_beam_search(self):
        pass

    @unittest.skip(reason="Chroma has special embeddings that can never be tied")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="Chroma has no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="Chroma has complex multi-component architecture")
    def test_initialization(self):
        pass

    @unittest.skip(reason="Chroma has complex multi-component architecture")
    def test_save_load(self):
        pass

    @unittest.skip(reason="Chroma has complex multi-component architecture")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Chroma has complex multi-component architecture")
    def test_save_load_fast_init_to_base(self):
        pass

    def _get_custom_4d_mask_test_data(self):
        """
        Overrides [ModelTesterMixin._get_custom_4d_mask_test_data] to handle third input_ids dimension.
        """
        # Sequence in which all but the last token is the same
        input_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]], device=torch_device, dtype=torch.int64)
        input_ids = input_ids.unsqueeze(-1).expand(-1, -1, self.model_tester.config["audio_num_codebooks"])
        position_ids = torch.tensor([[0, 1, 2, 3]] * 3, device=torch_device, dtype=torch.int64)

        # Combining common prefix with the unique ending tokens:
        input_ids_shared_prefix = torch.cat([input_ids[0][:-1], input_ids[:, -1]]).unsqueeze(0)

        # Creating a 4D mask where each of the last 3 tokens do not attend to each other.
        mask_shared_prefix = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 1, 0],
                        [1, 1, 1, 0, 0, 1],
                    ]
                ]
            ],
        )
        # inverting the attention mask
        mask_dtype = torch.float32
        min_dtype = torch.finfo(mask_dtype).min
        mask_shared_prefix = (mask_shared_prefix.eq(0.0)).to(dtype=mask_dtype, device=torch_device) * min_dtype

        # Creating a position_ids tensor. note the repeating figures in the end.
        position_ids_shared_prefix = torch.tensor([[0, 1, 2, 3, 3, 3]], device=torch_device, dtype=torch.int64)

        return input_ids, position_ids, input_ids_shared_prefix, mask_shared_prefix, position_ids_shared_prefix


@require_read_token
class ChromaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        # TODO: update with correct Chroma model checkpoint
        self.model_checkpoint = "/models/Qwencsm/Chroma/checkpoints/chroma_transformers_1119"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    @unittest.skip(reason="Chroma integration tests require actual model checkpoint")
    def test_chroma_model_integration_generate_text_input(self):
        """
        Tests the generated tokens match the ones from the original model implementation.
        """
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        # Prepare test conversation
        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        text = [ds[0]["text"]]
        audio = [ds[0]["audio"]["array"]]
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            },
        ]

        # Prepare prompt audio and text
        prompt_text = text
        prompt_audio = audio

        inputs = processor(
            conversations=conversations,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
            return_tensors="pt"
        ).to(torch_device)

        model = ChromaForConditionalGeneration.from_pretrained(self.model_checkpoint, device_map=torch_device)
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            use_cache=True,
            eos_token_id=0,
            pad_token_id=2050,
        )

        EXPECTED_OUTPUT_TOKENS = torch.tensor([[
            [799, 1803, 725, 1510, 583, 1689, 1692, 1044],
            [1209, 1412, 1658, 264, 639, 794, 1851, 140],
            [1037, 1716, 29, 1716, 26, 1244, 1069, 1249],
            [819, 277, 1875, 1267, 1601, 1529, 975, 1572],
            [125, 1982, 862, 474, 731, 1081, 542, 1107],
            [1516, 637, 1100, 1859, 1826, 2011, 1361, 326],
            [2008, 616, 722, 1267, 224, 1766, 1499, 1018],
            [1384, 1720, 271, 1569, 673, 542, 1221, 1997],
            [982, 371, 447, 1232, 1961, 1690, 921, 559],
            [1959, 1716, 1547, 454, 1994, 344, 1308, 1977],
            [1532, 1843, 211, 1653, 885, 623, 993, 1115],
            [1334, 1679, 1473, 55, 536, 1296, 387, 1050],
            [1985, 156, 834, 301, 1818, 1981, 1497, 585],
            [798, 75, 1038, 1653, 379, 6, 1402, 302],
            [150, 2033, 890, 1267, 1812, 1662, 921, 418],
            [150, 1967, 502, 560, 748, 961, 2030, 1268],
            [1547, 1438, 461, 46, 273, 1383, 2008, 68],
            [1259, 1663, 470, 769, 44, 1844, 611, 948],
            [1140, 10, 37, 1692, 876, 1233, 697, 469],
            [1494, 1803, 993, 264, 1031, 534, 402, 158],
            [778, 1138, 1064, 201, 1427, 1554, 435, 984],
            [1558, 369, 513, 802, 400, 395, 1997, 57],
            [1697, 471, 1739, 248, 1840, 86, 1388, 1279],
            [623, 1818, 1386, 1741, 956, 264, 1745, 1851],
            [635, 1478, 952, 676, 1961, 52, 1095, 395]
        ]])
        # TODO: Add expected output tokens once we have reference implementation
        torch.testing.assert_close(output_tokens.cpu(), EXPECTED_OUTPUT_TOKENS)

        # For now, just check that generation produces valid output
        self.assertIsNotNone(output_tokens)
        self.assertEqual(output_tokens.dim(), 3)  # [batch, seq_len, num_codebooks]

    @slow
    @require_torch_accelerator
    @unittest.skip(reason="Chroma integration tests require actual model checkpoint")
    def test_chroma_model_integration_generate_audio_input(self):
        """
        Tests the generated tokens match the ones from the original model implementation.
        """
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        # Prepare test conversation
        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        text = [ds[0]["text"]]
        audio = [ds[0]["audio"]["array"]]
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "text": audio},
                ],
            },
        ]

        # Prepare prompt audio and text
        prompt_text = text
        prompt_audio = audio

        inputs = processor(
            conversations=conversations,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
            return_tensors="pt"
        ).to(torch_device)

        model = ChromaForConditionalGeneration.from_pretrained(self.model_checkpoint, device_map=torch_device)
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            use_cache=True,
            eos_token_id=0,
            pad_token_id=2050,
        )

        EXPECTED_OUTPUT_TOKENS = torch.tensor([[
            [799, 1803, 725, 1510, 583, 1689, 1692, 1044],
            [1209, 1412, 1658, 264, 639, 794, 1851, 140],
            [1037, 1716, 29, 1716, 26, 1244, 1069, 1249],
            [819, 277, 1875, 1267, 1601, 1529, 975, 1572],
            [125, 1982, 862, 474, 731, 1081, 542, 1107],
            [1516, 637, 1100, 1859, 1826, 2011, 1361, 326],
            [2008, 616, 722, 1267, 224, 1766, 1499, 1018],
            [1384, 1720, 271, 1569, 673, 542, 1221, 1997],
            [982, 371, 447, 1232, 1961, 1690, 921, 559],
            [1959, 1716, 1547, 454, 1994, 344, 1308, 1977],
            [1532, 1843, 211, 1653, 885, 623, 993, 1115],
            [1334, 1679, 1473, 55, 536, 1296, 387, 1050],
            [1985, 156, 834, 301, 1818, 1981, 1497, 585],
            [798, 75, 1038, 1653, 379, 6, 1402, 302],
            [150, 2033, 890, 1267, 1812, 1662, 921, 418],
            [150, 1967, 502, 560, 748, 961, 2030, 1268],
            [1547, 1438, 461, 46, 273, 1383, 2008, 68],
            [1259, 1663, 470, 769, 44, 1844, 611, 948],
            [1140, 10, 37, 1692, 876, 1233, 697, 469],
            [1494, 1803, 993, 264, 1031, 534, 402, 158],
            [778, 1138, 1064, 201, 1427, 1554, 435, 984],
            [1558, 369, 513, 802, 400, 395, 1997, 57],
            [1697, 471, 1739, 248, 1840, 86, 1388, 1279],
            [623, 1818, 1386, 1741, 956, 264, 1745, 1851],
            [635, 1478, 952, 676, 1961, 52, 1095, 395]
        ]])

        # TODO: Add expected output tokens once we have reference implementation
        torch.testing.assert_close(output_tokens.cpu(), EXPECTED_OUTPUT_TOKENS)

        # For now, just check that generation produces valid output
        self.assertIsNotNone(output_tokens)
        self.assertEqual(output_tokens.dim(), 3)  # [batch, seq_len, num_codebooks]

    @slow
    @require_torch_accelerator
    @unittest.skip(reason="Chroma integration tests require actual model checkpoint")
    def test_chroma_model_integration_generate_batched_text_inputs(self):
        """
        Test the generated tokens with batched inputs.
        """
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        texts = ds[:2]["text"]
        audios = [ds[0]["audio"]["array"], ds[1]["audio"]["array"]]

        conversations = [
            # 第一个样本
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": texts[0]},
                    ],
                },
            ],
            # 第二个样本
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": texts[1]},
                    ],
                },
            ],
        ]

        prompt_texts = [texts[0], texts[1], ]

        prompt_audios = [audios[0], audios[1], ]

        inputs = processor(
            conversations=conversations,
            prompt_text=prompt_texts,
            prompt_audio=prompt_audios,
            return_tensors="pt"
        ).to(torch_device)

        model = ChromaForConditionalGeneration.from_pretrained(self.model_checkpoint, device_map=torch_device)
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            use_cache=True,
            eos_token_id=0,
            pad_token_id=2050,
        )

        EXPECTED_OUTPUT_TOKENS = torch.tensor([
            [[1952, 545, 873, 201, 1218, 1180, 1755, 537],
             [296, 1284, 1307, 301, 1595, 320, 812, 1082],
             [117, 714, 24, 201, 438, 1382, 737, 537],
             [683, 409, 862, 1654, 468, 1467, 1570, 1702],
             [329, 366, 878, 1289, 695, 328, 90, 1158],
             [1078, 1803, 957, 1337, 1468, 1266, 1189, 1505],
             [1560, 1803, 1126, 964, 154, 415, 173, 1997],
             [747, 2004, 30, 799, 1229, 1408, 1799, 900],
             [1566, 1843, 2005, 374, 1449, 1033, 1057, 835],
             [32, 463, 1856, 1918, 890, 685, 1082, 1439],
             [845, 386, 1007, 1065, 1470, 1889, 1809, 186],
             [257, 1034, 1879, 1990, 501, 56, 524, 1460],
             [683, 463, 301, 1486, 1447, 936, 1739, 981],
             [1508, 314, 441, 40, 583, 1328, 1085, 117],
             [915, 803, 1181, 617, 1193, 240, 1596, 1607],
             [1828, 1453, 538, 1453, 17, 1611, 688, 1634],
             [1733, 564, 1577, 314, 1575, 274, 481, 1246],
             [329, 348, 1414, 1210, 1403, 264, 1471, 1461],
             [1972, 1529, 1645, 1803, 655, 1806, 1429, 1586],
             [839, 1488, 135, 657, 1214, 46, 1029, 158],
             [1628, 839, 391, 537, 1010, 1193, 476, 1933],
             [185, 1052, 1577, 1868, 911, 837, 1896, 733],
             [548, 512, 1869, 1738, 854, 1819, 996, 1704],
             [1967, 1843, 1288, 200, 684, 6, 706, 1191],
             [1945, 1843, 1288, 524, 855, 1762, 1402, 1115]],

            [[1698, 991, 1368, 1697, 1403, 1015, 1140, 1670],
             [109, 176, 1195, 1880, 1266, 1620, 187, 1058],
             [1670, 1474, 1638, 277, 163, 1920, 241, 1069],
             [1670, 244, 84, 1447, 1706, 1925, 239, 743],
             [1877, 388, 1273, 282, 911, 1057, 1663, 708],
             [1039, 786, 711, 177, 638, 76, 1386, 1063],
             [1640, 1515, 1626, 142, 306, 555, 160, 1744],
             [541, 243, 1697, 164, 267, 555, 976, 1648],
             [448, 243, 1559, 546, 481, 1030, 666, 1744],
             [448, 243, 1559, 546, 481, 1030, 976, 1744],
             [84, 243, 1559, 546, 481, 1030, 976, 1744],
             [1850, 243, 1559, 546, 481, 1030, 976, 1744],
             [752, 243, 1559, 546, 481, 1030, 976, 1744],
             [1850, 243, 1559, 546, 481, 1030, 976, 1744],
             [752, 243, 1559, 546, 481, 1030, 825, 1744],
             [481, 243, 1559, 546, 481, 1030, 825, 1744],
             [481, 243, 1559, 546, 481, 1030, 825, 1744],
             [481, 243, 1559, 164, 267, 1572, 976, 1744],
             [1926, 243, 1559, 546, 1736, 1572, 1978, 1744],
             [481, 243, 1559, 164, 267, 1572, 976, 1744],
             [1926, 243, 1178, 546, 267, 1030, 1978, 1744],
             [1926, 243, 1178, 1348, 1335, 1572, 666, 1744],
             [1926, 243, 1178, 546, 481, 1030, 666, 1744],
             [1926, 243, 1178, 1348, 1335, 1572, 666, 1744],
             [1926, 243, 1178, 546, 481, 1030, 666, 1744]]
        ])
        torch.testing.assert_close(output_tokens.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
    @unittest.skip(reason="Chroma integration tests require actual model checkpoint")
    def test_chroma_model_integration_generate_batched_audio_inputs(self):
        """
        Test the generated tokens with batched inputs.
        """
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        texts = ds[:2]["text"]
        audios = [ds[0]["audio"]["array"], ds[1]["audio"]["array"]]

        conversations = [
            # 第一个样本
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audios[0]},
                    ],
                },
            ],
            # 第二个样本
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audios[1]},
                    ],
                },
            ],
        ]

        prompt_texts = [texts[0], texts[1], ]

        prompt_audios = [audios[0], audios[1], ]

        inputs = processor(
            conversations=conversations,
            prompt_text=prompt_texts,
            prompt_audio=prompt_audios,
            return_tensors="pt"
        ).to(torch_device)

        model = ChromaForConditionalGeneration.from_pretrained(self.model_checkpoint, device_map=torch_device)
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            eos_token_id=0,
            pad_token_id=2050,
        )

        EXPECTED_OUTPUT_TOKENS = torch.tensor([
            [[975, 1803, 423, 1267, 1546, 385, 46, 1314],
             [778, 251, 1982, 1414, 193, 286, 1808, 2033],
             [1606, 533, 56, 1510, 764, 991, 1061, 191],
             [837, 22, 1086, 1900, 928, 1189, 1409, 1268],
             [117, 22, 1086, 159, 1485, 1131, 1289, 1748],
             [994, 1052, 236, 542, 1525, 253, 678, 257],
             [603, 1591, 782, 1883, 748, 625, 1578, 630],
             [847, 533, 285, 743, 639, 1770, 1253, 1851],
             [1109, 533, 548, 627, 1369, 1435, 1474, 1546],
             [1256, 394, 273, 1068, 211, 1383, 816, 1950],
             [2000, 1052, 634, 1188, 1168, 1714, 1166, 298],
             [2042, 316, 1697, 114, 1015, 555, 428, 1606],
             [541, 316, 1178, 142, 267, 347, 1978, 1665],
             [448, 243, 1697, 142, 481, 347, 666, 2008],
             [84, 243, 1697, 546, 1736, 1030, 1978, 1744],
             [1850, 243, 1697, 164, 267, 1572, 825, 1744],
             [752, 243, 783, 142, 481, 1030, 666, 1744],
             [752, 243, 783, 142, 481, 1030, 666, 1744],
             [752, 243, 783, 142, 481, 1030, 666, 1744],
             [752, 243, 783, 142, 481, 1030, 666, 1744],
             [752, 243, 783, 142, 481, 1030, 666, 1744],
             [752, 243, 1178, 546, 267, 555, 825, 1648],
             [481, 243, 1178, 546, 267, 1030, 825, 1744],
             [1926, 243, 1178, 546, 267, 1030, 825, 1744],
             [1926, 243, 1178, 546, 267, 1030, 825, 1744]],

            [[1182, 398, 1265, 743, 666, 344, 1627, 1600],
             [301, 1161, 114, 1366, 972, 770, 310, 177],
             [561, 1486, 233, 524, 650, 6, 415, 1115],
             [1506, 1855, 1530, 1987, 970, 954, 909, 893],
             [292, 1832, 1868, 1987, 1183, 648, 356, 67],
             [343, 1985, 582, 2001, 1937, 417, 1568, 211],
             [1018, 1185, 923, 467, 886, 150, 313, 1261],
             [390, 723, 463, 1975, 1203, 1177, 804, 1524],
             [854, 343, 312, 1564, 1222, 867, 1305, 862],
             [98, 127, 1265, 854, 1596, 1848, 852, 170],
             [98, 896, 200, 494, 972, 1920, 563, 829],
             [646, 1929, 1088, 1928, 204, 532, 1247, 573],
             [1592, 787, 62, 494, 1302, 1723, 842, 200],
             [188, 1973, 1585, 545, 657, 1371, 1001, 1734],
             [918, 631, 2002, 1466, 1560, 933, 1745, 606],
             [1726, 1918, 1668, 356, 1914, 75, 859, 451],
             [1622, 659, 1471, 1053, 1702, 1136, 1741, 578],
             [890, 909, 1660, 1893, 1481, 361, 1199, 1403],
             [1776, 496, 1963, 404, 1732, 180, 1153, 1997],
             [704, 1260, 171, 2028, 744, 239, 71, 432],
             [225, 482, 54, 2001, 1126, 1219, 1956, 245],
             [264, 1396, 1812, 151, 1390, 29, 1141, 1847],
             [544, 1415, 938, 29, 1486, 898, 971, 278],
             [1005, 461, 84, 45, 1485, 1092, 447, 1129],
             [1914, 21, 1643, 817, 372, 226, 1768, 473]]
        ])
        torch.testing.assert_close(output_tokens.cpu(), EXPECTED_OUTPUT_TOKENS)

