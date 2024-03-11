import torch
import os
import time

def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    seq_groups  = None,
    generators  = None,
) -> torch.Tensor:
    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        # This allows us to do sampling with replacement by creating
        # num_samples copies of each row in the tensor, and then
        # batch sampling the resulting tensor.
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
                                             -1, probs.shape[1])
    q = torch.empty_like(probs)
    if seq_groups is None:
        q.exponential_()
    else:
        sample_idx = 0
        for (seq_ids, _), generator in zip(seq_groups, generators):
            next_sample_idx = sample_idx + len(seq_ids) * num_samples
            q[sample_idx:next_sample_idx].exponential_(generator=generator)
            sample_idx = next_sample_idx
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)

def _prune_hidden_states(hidden_states, sampling_metadata):
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    selected_token_indices=torch.tensor([ 67, 417], device='cuda:0')
    return hidden_states.index_select(0, selected_token_indices)

def _sample(probs, logprobs, sampling_metadata):
    sampling_results = []
    for j in range(10):
        sampling_results.append( _multinomial(probs, 1))
    return sampling_results

class Sampler(torch.nn.Module):
    def __init__(self, vocab_size, org_vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.org_vocab_size = org_vocab_size

    def _get_logits(self, hidden_states, embedding, embedding_bias):
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias:
            logits += embedding_bias

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[:, :self.org_vocab_size]
        return logits


    def forward(self, embedding, hidden_states, sampling_metadata=None, embedding_bias=None):
        hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)
        logits = self._get_logits(hidden_states, embedding, embedding_bias)
        assert logits is not None
        _, vocab_size = logits.shape
        ##logits = _apply_logits_processors(logits, sampling_metadata)
        sampling_tensors = torch.ones((2, 32000)).cuda()
        logits.div_(sampling_tensors.unsqueeze_(dim=1).view(2, 32000))
        ## add topp and topk
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        sampling_results = _sample(probs, logprobs, sampling_metadata)        
        return sampling_results


embedding_tensor = torch.randn(32000, 4096, dtype=torch.half, device='cuda')
hidden_states = torch.randn(2, 209, 4096, dtype=torch.half, device='cuda')
vocab_size = 32000
org_vocab_size = 32000
sampler_model = Sampler(vocab_size, org_vocab_size)

outputs = sampler_model(embedding_tensor, hidden_states)
outputs = sampler_model(embedding_tensor, hidden_states)
print ("OK: finished eager mode.")

torch.cuda.synchronize()
start_time = time.time()
for j in range(1000):
    outputs = sampler_model(embedding_tensor, hidden_states)
torch.cuda.synchronize()
end_time = time.time()
print ("INFO: Sampler eager time: {}".format(end_time - start_time)) 


model_compile = torch.compile(sampler_model, backend="inductor")

## warmup
for j in range(3):
    outputs_comp = model_compile(embedding_tensor, hidden_states)

## timing.
torch.cuda.synchronize()
start_time = time.time()
for j in range(1000):
    outputs_comp = model_compile(embedding_tensor, hidden_states)
torch.cuda.synchronize()
end_time = time.time()
print ("INFO: Sampler torch compile time: {}".format(end_time - start_time))





       
'''

----- CHAI: hidden_states type : torch.cuda.HalfTensor
CHAI: Sampler forward -----
CHAI: embedding size: torch.Size([32000, 4096]), dtype: torch.cuda.HalfTensor
CHAI: hidden_states size : torch.Size([2, 209, 4096]), dtype: torch.cuda.HalfTensor
CHAI: sampling_metadata : SamplingMetadata(seq_groups=[([0], SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_pena
CHAI: embeding_bias is None

'''
        
#class Sampler(torch.nn.Module):
#    def __init__(self, vocab_size, org_vocab_size):
#        super().__init__()
#        self.vocab_size = vocab_size
#        self.org_vocab_size = org_vocab_size
#
    
#probs = torch.randn(256, 32000).cuda()
#print (probs.size())
#num_samples = 1
#output1 = _multinomial(probs, 1)
#
#_multinomial_func = torch.compile(_multinomial, backend="inductor")
#output2 = _multinomial_func(probs, 1)
#
### eager time.
#import time
#torch.cuda.synchronize()
#start_time = time.time()
#for i in range(1000):
#    output1 = _multinomial(probs, 1)
#torch.cuda.synchronize()
#end_time = time.time()
#
#print ("Eager time for multinomial: {}".format(end_time - start_time))
#
#torch.cuda.synchronize()
#start_time = time.time()
#for i in range(1000):
#    output2 = _multinomial_func(probs, 1)
#torch.cuda.synchronize()
#end_time = time.time()
#
#print ("Compile time for multinomial: {}".format(end_time - start_time))
#
