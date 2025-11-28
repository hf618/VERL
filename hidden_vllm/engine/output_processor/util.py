from typing import List
from typing import Sequence as GenericSequence
from typing import Union
import torch
from hidden_vllm.sequence import PoolerOutput, SamplerOutput, SequenceGroupOutput

##########changed!!!!!!####################
def create_output_by_sequence_group(
        outputs: GenericSequence[Union[SamplerOutput, PoolerOutput]],
        num_seq_groups: int) -> List[List[SequenceGroupOutput]]:
    """Helper method which transforms a 2d list organized by
    [step][sequence group] into [sequence group][step].
    """
    output_by_sequence_group: List[List[SequenceGroupOutput]] = [
        [] for _ in range(num_seq_groups)
    ]
    
    for step in outputs:
        for i, sequence_group_output in enumerate(step): # micro bs
            # Use torch.stack instead of a manual loop

            for j in range(len(sequence_group_output.samples)): # rollouts n
                #  
                if hasattr(step, 'hidden_states_decode'):
                    if step.hidden_states_decode[0].shape[0] == len(sequence_group_output.samples) * len(step.outputs): # If rollouts have already branched into n * micro batch size
                        hidden_states_tensor = torch.stack([hidden_state[i+j] for hidden_state in step.hidden_states_decode])
                    else:
                        hidden_states_tensor = torch.stack([hidden_state[i] for hidden_state in step.hidden_states_decode])
                    sequence_group_output.samples[j].hidden_states_decode = hidden_states_tensor # layers * dim 
            
            
            prefill_states = getattr(step, 'hidden_states_prefill', None)
            if prefill_states is not None:
                # prefill_states[0] has shape (micro_bs, seq_len, num_layers, hidden_dim)
                # Directly fetch the tensor for the corresponding batch index
                sequence_group_output.hidden_states_prefill = [prefill_states[0][i]]
            else:
                sequence_group_output.hidden_states_prefill = None
            output_by_sequence_group[i].append(sequence_group_output)
            
     
    return output_by_sequence_group
