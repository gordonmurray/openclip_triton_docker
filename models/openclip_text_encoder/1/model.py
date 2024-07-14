import torch
import open_clip
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_name = 'ViT-L-14'
        self.model_path = '/model_cache/laion--CLIP-ViT-L-14-laion2B-s32B-b82K/model.bin'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.model_path,
            jit=False,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get the input text tensor
            input_text = pb_utils.get_input_tensor_by_name(request, 'text_encoder_input').as_numpy()
            input_text = input_text[0].decode('utf-8')  # Decode bytes to string

            # Tokenize the input text
            input_tokens = self.tokenizer([input_text])
            input_tensors = torch.tensor(input_tokens).to(self.device)

            with torch.no_grad():
                # Get the text embeddings from the model
                output_embeddings = self.model.encode_text(input_tensors).cpu().numpy()[0]

            # Create the output tensor
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor('text_encoder_output', output_embeddings)]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass
