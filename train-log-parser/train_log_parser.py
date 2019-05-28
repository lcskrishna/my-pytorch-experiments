import os
import sys
import argparse

def read_log_file(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()
    
    useful_lines = []
    validation_lines = []
    for i in range(len(lines)):
        line = lines[i].rstrip()
        if "Epoch:" in line and ("UserWarning" not in line):
            useful_lines.append(line)
        if "* Prec@1" in line:
            validation_lines.append(line)

    return useful_lines, validation_lines

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
    #print (len(unique_epochs))
    #print (len(epoch_values))
    #print (len(loss_values))
    for i in range(len(unique_epochs)):
        epoch_num = unique_epochs[i]
        #print ("INFO: Current epoch number is {}".format(epoch_num))
        tot_loss = 0
        count = 0
        for j in range(len(epoch_values)):
            if epoch_num == epoch_values[j]:
                tot_loss = tot_loss + loss_values[j]
                count = count + 1
        avg_loss = tot_loss / count
        losses.append(avg_loss)
        #print (count)
        #print (avg_loss)

    fs = open(output_prefix + ".csv", "w")
    for j in range(len(losses)):
        line = str(j) + "," + str(losses[j]) + "\n"
        fs.write(line)

    fs.close()
    print ("OK: generated csv for losses and epochs")

def generate_prec1_prec5_csv(epoch_values, prec1_values, prec5_values, output_prefix):
    unique_epochs = list(set(epoch_values))
    prec1 = []
    prec5 = []
    #print (len(unique_epochs))
    #print (len(prec1_values))
    #print (len(prec5_values))
    #print (len(epoch_values))
    for i in range(len(unique_epochs)):
        epoch_num = unique_epochs[i]
        #print ("INFO: Current epoch is {}".format(epoch_num))
        total_prec1 = 0
        total_prec5 = 0
        count = 0
        for j in range(len(epoch_values)):
            if epoch_num == epoch_values[j]:
                total_prec1 = total_prec1 + prec1_values[j]
                total_prec5 = total_prec5 + prec5_values[j]
                count = count + 1
        avg_top1 = total_prec1 / count
        avg_top5 = total_prec5 / count
        prec1.append(avg_top1)
        prec5.append(avg_top5)

    fs = open(output_prefix + "_top1_top5.csv", "w")
    for j in range(len(prec1)):
        line = str(j) + "," + str(prec1[j]) + "," + str(prec5[j]) + "\n"
        fs.write(line)

    fs.close()
    print ("OK: generated csv for top1 and top5 for each epochs.")

def generate_test_prec1_prec5_csv(validation_lines, output_prefix):
    print ("INFO: total number of validation results obtained is : {}".format(len(validation_lines)))
    prec1_values = []
    prec5_values = []
    for i in range(len(validation_lines)):
        line = validation_lines[i].rstrip()
        info = line.split(" ")
        prec1_values.append(info[3])
        prec5_values.append(info[5])
    
    fs = open(output_prefix + "_test_top1_top5.csv", "w")
    for j in range(len(prec1_values)):
        line = str(j) + "," + str(prec1_values[j]) + "," + str(prec5_values[j]) + "\n"
        fs.write(line)
    fs.close()
    print ("OK: generated csv for top1 and top5 for test datasets")
        
 
def main():
    log_file = os.path.abspath(args.log_file)
    output_prefix = args.output_prefix
    lines, validation_lines = read_log_file(log_file)
    epoch_values, loss_values, prec1_values, prec5_values = extract_info(lines)
    generate_training_loss_csv(epoch_values, loss_values, output_prefix)
    generate_prec1_prec5_csv(epoch_values, prec1_values, prec5_values, output_prefix)
    generate_test_prec1_prec5_csv(validation_lines, output_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str, required=True, help='Training log file.')
    parser.add_argument('--output-prefix', type=str, required=False, default = 'net', help='Output prefix.')

    args = parser.parse_args()
    main()

