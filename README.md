# torchvggish-gpu
Re-Implementation of Google Research's VGGish model used for extracting audio features using Pytorch with GPU support.

A re-implementation of [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset)<sup>[1]</sup>, 
a feature embedding frontend for audio classification models, using Pytorch with GPU support. This code is fully based on [torchvggish](https://github.com/harritaylor/torchvggish)<sup>[2]</sup>.


## Usage

```python
import torch
from torchvggish_gpu import vggish
import vggish_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# For GPU support, the device must be cuda

embedding_model = vggish()
embedding_model.to(device)
embedding_model.eval()
example = vggish_input.wavfile_to_examples("bus_chatter.wav")
example = example.to(device)
audio_embeddings = embedding_model.forward(example)
```

<hr>
[1]  S. Hershey et al., ‘CNN Architectures for Large-Scale Audio Classification’,
    in International Conference on Acoustics, Speech and Signal Processing (ICASSP),2017.
    Available: https://arxiv.org/abs/1609.09430, https://ai.google/research/pubs/pub45611

[2] Harri Taylor et al., ‘Pytorch port of Google Research's VGGish model used for extracting audio features’,
    v0.1, Sep 27, 2019. Available: https://github.com/harritaylor/torchvggish/releases/tag/v0.1
