import numpy as np
import json
import os 

class Sequence:
    '''
    Class for individual sequence in game
    '''
    def __init__(self, sequence) -> None:
        '''
        -sequence : single sequence from game
        '''
        self.start_time = sequence[0][2]
        self.end_time = sequence[-1][2]
        self.total_time = self.start_time - self.end_time
        self.player_ball_positions = [s[-1] for s in sequence]
        self.total_pos_moments = len(self.player_ball_positions)
        self.contains_ball = True

        #check if ball position is in data
        for x in self.player_ball_positions:
            if len(x) != 11:
                self.contains_ball = False
                break
        

    def downsample(self, sampling_rate):
        '''
        Downsample positions if needed 
        Data set is sampled 25 times per second. Can downsample to decrease amount of data and increase time in between data points
        '''
        skip = int(25 / sampling_rate)
        return [self.player_ball_positions[i] for i in range(0,self.total_pos_moments, skip)]
        


#create dataset from sequences
class BallerDataset:
    '''
    Class for dataset 
    '''
    def __init__(self, data_path, seq_len, ball_bins_width ,sampling_rate=25):
        '''
        - data_path : directory of data for all games, (games must be json files)
        - seq_len : how long to make training sequences (label is for seq_len+1)
        - ball_bins_width : width of the ball bins trajectory output
        - sampling_rate : samples per second (Hz)
        '''
        self.data_path = data_path
        self.seq_len = seq_len + 1
        self.sampling_rate = sampling_rate
        self.ball_bins_width = ball_bins_width
      

    def createData(self,):
        '''
        Create all data
        
        Returns:
            -all_data_list: list of game data each element in list is np array(samples, seq_len, 11,3)
            -unpadded_seq_lens: list of unpadded sequence lengths for each game
                                 Each element of size (samples, 1)
        '''
        games_names = os.listdir(self.data_path)

        all_data_list, all_unpadded_seq_lens = [],[]

        for game_name in games_names:
            game = self.readGame(os.path.join(self.data_path, game_name))
            sequences = self.getSequences(game)
            sequences = self.processSequences(sequences)
            positions, unpadded_seq_lens = self.extractPositions(sequences)
            all_data_list.append(positions)
            all_unpadded_seq_lens.append(unpadded_seq_lens)

        #all_data = np.concatenate(all_data_list, axis=0)
        return all_data_list, all_unpadded_seq_lens

    def readGame(self, game_path):
        f = open(game_path)
        game = json.load(f)
        return game
    
    def getSequences(self,game):
        sequences = []
        start_end_times= set()
      
        for event in game['events']:
            if len(event['moments'] )>0:
                seq = Sequence(event['moments'])

                #dont add sequences without ball position
                if not seq.contains_ball:
                    continue

                #only add sequences that are not identical
                if (seq.start_time, seq.end_time) not in start_end_times:
                    sequences.append(seq)
                    start_end_times.add((seq.start_time, seq.end_time))
        return sequences

    def processSequences(self, sequences):
        '''
        Downsample and break up sequences longer than seq_len
        '''
        processed= []

        for seq in sequences:
            downsampled_positions = seq.downsample(self.sampling_rate)

            #split into chunks if is is longer than seq_len
            n = self.seq_len
            chunks = [downsampled_positions[i:i+n] for i in range(0, len(downsampled_positions),n)]
            processed += chunks

        return processed

    def extractPositions(self, sequences):
        '''
        Turn sequences into numpy array data of form (samples, seq_len, 11,3)

        Returns:
            - data: (samples, seq_len, 11,3)
            - sequence_unpadded_lengths: length of oringial unpadded sequence
        '''
        data = np.zeros((len(sequences), self.seq_len, 11,3))
        sequence_unpadded_lengths = []
        for i,seq in enumerate(sequences):
            seq_np =np.array(seq)
            sequence_unpadded_lengths.append(seq_np.shape[0])

            #no padding required
            if seq_np.shape[0] == self.seq_len:
                data[i] = seq_np[:,:,-3:]

            #padding required
            else:
                s = seq_np[:,:,-3:]
                dim = seq_np.shape[0]
                padding_dim = self.seq_len - dim

                npad = ((0,padding_dim),(0,0),(0,0))
                padded = np.pad(s, pad_width=npad, mode='constant', constant_values=0)
                data[i] = padded
        return data, sequence_unpadded_lengths
    
    
    '''
    def getBallLabels(self, data):
        
        Get all labels for all time steps
        - data: numpy array of size (samples, seq_len, 11,3)

        Returns:
            binned_indices: binned trajectory of balls (samples, seq_len-1)

        
        
        balls = data[:,:,0,:]
      
        balls_diff = np.diff(balls, axis=1)     #shape (samples, seq_len-1,3)
        
        #create function to convert ball diffs to one hot bin labels
        binned_indices = self.diff_to_bins(balls_diff, self.ball_bins_width) #shape (samples, seq_len-1, width**3)

        return binned_indices
    '''

    def getBallLabels(self,all_data_list, unpadded_lengths_list):
        '''
        Get ball labels for last time step
        - all_data_list: list of data each element of size(samples, seq_len, 11,3)
        - unpadded_lengths_list: list of unpadded sequence lengths for each game
                                 Each element of size (samples, 1)

        Returns:
            -last_labels: (samples, 3)
        '''

        last_labels_list = []
        for i in range(len(all_data_list)):
            data = all_data_list[i]
            unpadded_lengths = unpadded_lengths_list[i]
            balls = data[:,:,0,:]
            balls_diff = np.diff(balls, axis=1)     #shape (samples, seq_len-1,3)

            unpadded_lengths = np.array(unpadded_lengths)
            last_labels = balls_diff[np.arange(len(unpadded_lengths)), unpadded_lengths -2] #shape (samples, 3)
            last_labels_list.append(last_labels)
            #last_labels = np.expand_dims(last_labels, axis= 1)

#            binned_indices = self.diff_to_bins(last_labels, self.ball_bins_width)
 #           binned_indices_list.append(binned_indices)

#        binned_indices = np.concatenate(binned_indices_list, axis=0)
        return np.concatenate(last_labels_list)




    def diff_to_bins(self, diffs, width):
        '''
        - diffs: ball pos diffs (samples, n, 3)
        - width: size of the 3d ball prediction bins(width x width x width)

        Returns:
            box_indices: of shape (samples, n)
        '''
        center_ind = width*width*width // 2
        max_diff_per_dimension = width // 2 

        #convert diffs to ints and bound values to be within box(in case ball moves super far super fast) 
        diffs_bounded = diffs.astype(int)
        diffs_bounded[diffs_bounded > max_diff_per_dimension] = max_diff_per_dimension
        diffs_bounded[diffs_bounded < -1*max_diff_per_dimension] = -1* max_diff_per_dimension

        x_offsets = diffs_bounded[:,:,0]
        y_offsets = diffs_bounded[:,:,1]
        z_offsets = diffs_bounded[:,:,2]

        box_indices = center_ind + x_offsets + (y_offsets * width) + (z_offsets * width*width)  #(samples, n)
        return box_indices
      





