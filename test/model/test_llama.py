import os
import sys
import unittest
from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestLlamaInfer(unittest.TestCase):

    def test_llama_infer(self):
        test_model_inference(world_size=1,
                             model_dir="/path/to/llama-7b",
                             model_class='lightllm.models.llama.model.LlamaTpPartModel',
                             batch_size=20,
                             input_len=1024,
                             output_len=1024,
                             use_patch=False,
                             )
        return


if __name__ == '__main__':
    unittest.main()
