import torch
import open_clip
from PIL import Image
import triton_python_backend_utils as pb_utils
import io
import base64

class TritonPythonModel:
    def initialize(self, args):
        self.model_name = 'ViT-L-14'
        self.model_path = '/model_cache/laion--CLIP-ViT-L-14-laion2B-s32B-b82K/model.bin'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.model_path,
            jit=False,
            device=self.device
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get the input image tensor
            input_image = pb_utils.get_input_tensor_by_name(request, 'image_encoder_input').as_numpy()[0]
            image_data = base64.b64decode(input_image)
            image = Image.open(io.BytesIO(image_data))

            # Preprocess the image
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Get the image embeddings from the model
                output_embeddings = self.model.encode_image(image_tensor).cpu().numpy()[0]

            # Create the output tensor
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor('image_encoder_output', output_embeddings)]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass
