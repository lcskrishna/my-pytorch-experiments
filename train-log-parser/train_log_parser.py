import os
import sys
import argparse

def read_log_file(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()
    
    useful_lines = []
    for i in range(len(lines)):
        line = lines[i].rstrip()
        if "Epoch:" in line and ("UserWarning" not in line):
            useful_lines.append(line)

    return useful_lines

def extract_info(lines):

    loss_values = []
    prec1_values = []
    epoch_values = []
    prec5_values = []

    for i in range(len(lines)):
        line = lines[i]
        #print (line)
        values = line.split(' ')
        epoch = values[1].split('][')[0].replace('[', '')
        epoch_values.append(int(epoch))
        loss = float(values[6])
        loss_values.append(loss)
        prec1 = float(values[8])
        prec1_values.append(prec1)
        prec5 = float(values[10])
        prec5_values.append(prec5)
    
    return epoch_values, loss_values, prec1_values, prec5_values        

def generate_training_loss_csv(epoch_values, loss_values, output_prefix):
    
    unique_epochs = list(set(epoch_values))
    losses = []
    print (len(unique_epochs))
    print (len(epoch_values))
    print (len(loss_values))
    for i in range(len(unique_epochs)):
        epoch_num = unique_epochs[i]
        print ("INFO: Current epoch number is {}".format(epoch_num))
        tot_loss = 0
        count = 0
        for j in range(len(epoch_values)):
            if epoch_num == epoch_values[j]:
                tot_loss = tot_loss + loss_values[j]
                count = count + 1
        avg_loss = tot_loss / count
        losses.append(avg_loss)
        print (count)
        print (avg_loss)

    fs = open(output_prefix + ".csv", "w")
    for j in range(len(losses)):
        line = str(j) + "," + str(losses[j]) + "\n"
        fs.write(line)

    fs.close()
    print ("OK: generated csv for losses and epochs")
 
def main():
    log_file = os.path.abspath(args.log_file)
    output_prefix = args.output_prefix
    lines = read_log_file(log_file)
    epoch_values, loss_values, prec1_values, prec5_values = extract_info(lines)
    generate_training_loss_csv(epoch_values, loss_values, output_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str, required=True, help='Training log file.')
    parser.add_argument('--output-prefix', type=str, required=False, default = 'net', help='Output prefix.')

    args = parser.parse_args()
    main()

