from glob import glob
import json
import numpy as np
import torch
import traceback

def load_game(file):
    with open(file, 'r') as f:
        return json.load(f)
    
def assertSequences(input_sequences, output_sequences, input_sequence_length, output_sequence_length):
    for iseq, oseq in zip(input_sequences, output_sequences):
        assert iseq.shape[0] == oseq.shape[0] and iseq.shape[0] != 0
        assert iseq.shape[1] == input_sequence_length
        assert oseq.shape[1] == output_sequence_length
        assert iseq.shape[2] == 11
        assert iseq.shape[3] == oseq.shape[2] and iseq.shape[3] == 3
        
def createSequences(game, input_sequence_length, output_sequence_length):
    total_sequence_length = input_sequence_length + output_sequence_length
    start_end_times = set()

    input_sequences = []
    output_sequences = []

    for event in game['events']:
        moments = event['moments']
        if len(moments) > 0:
            start_time, end_time = moments[0][2], moments[-1][2]
            if (start_time, end_time) in start_end_times:
                # Don't include repeat data
                continue
            start_end_times.add((start_time, end_time))

            positions = np.array([moment[-1] for moment in moments if len(moment[-1]) == 11])[...,-3:]
            
            theInput = [positions[i:i+input_sequence_length] for i in range(0, positions.shape[0]-total_sequence_length+1, input_sequence_length) if positions[i:i+input_sequence_length].shape[0] == input_sequence_length]
            theOutput = [positions[i:i+output_sequence_length,0] for i in range(input_sequence_length, positions.shape[0], input_sequence_length) if positions[i:i+output_sequence_length].shape[0] == output_sequence_length]

            if len(theInput) != 0 and len(theOutput) != 0:
                input_sequences.append(np.array(theInput)) # (moments, ballplayer, xyz)
                output_sequences.append(np.array(theOutput)) # (moments, xyz) of ball
                
            
    assertSequences(input_sequences, output_sequences, input_sequence_length, output_sequence_length)

    input_sequences = np.concatenate(input_sequences, axis=0) # (batch, horizon, ballplayer, xyz)
    output_sequences = np.concatenate(output_sequences, axis=0) # (batch, horizon, xyz) of ball
    
    return input_sequences, output_sequences

def file2Arr(file, input_sequence_length, output_sequence_length):
    game = load_game(file)
    try:
        input_seq, output_seq = createSequences(game, input_sequence_length, output_sequence_length)
    except Exception:
        print(f'Issue with {file}')
        traceback.print_exc()
        return

    with open(file.rstrip('.json')+'_in.pt', 'wb') as fin, open(file.rstrip('.json')+'_out.pt', 'wb') as fout:
        torch.save(torch.from_numpy(input_seq), fin)
        torch.save(torch.from_numpy(output_seq), fout)

if __name__ == '__main__':
    input_sequence_length, output_sequence_length = 25, 15
    for file in glob('./test/*.json'):
        file2Arr(file, input_sequence_length, output_sequence_length)