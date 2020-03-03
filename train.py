# https://github.com/spro/practical-pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import random
import string

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--n_epochs', type=int, default=2)
argparser.add_argument('--max_iters', type=int, default=0)

argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--hidden_size', type=int, default=256)
argparser.add_argument('--chunk_len', type=int, default=50)

argparser.add_argument('--all_rand', type=bool, default=False)
argparser.add_argument('--epoch_random', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.002)

argparser.add_argument('--print_every', type=int, default=1)
argparser.add_argument('--sample_every', type=int, default=0)
argparser.add_argument('--log_sample', type=int, default=50)
argparser.add_argument('--log_every', type=int, default=10)
argparser.add_argument('--save_every', type=int, default=100)

argparser.add_argument('--load', type=str, default='')
argparser.add_argument('--cuda', type=bool, default=False)
args = argparser.parse_args()

file, file_len = read_file(args.filename)
all_rand = args.all_rand
iters = math.floor(file_len / args.chunk_len)
if args.max_iters < iters and args.max_iters != 0:
    iters = args.max_iters
    all_rand = True
"""
def random_training_set(chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    chunk = file[start_index:end_index]
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target
"""

def get_chunk(start, chunk_len, random_chunk):
    if random_chunk:
        start_index = random.randint(0, file_len - chunk_len)
    else:
        start_index = (start - 1) * chunk_len
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

def training_set(chunk):
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

decoder = RNN(n_characters, args.hidden_size, n_characters, args.n_layers)
if args.cuda: decoder = decoder.cuda()
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

ep_start = 1
iter_start = 1

if args.load != '':
    print('Loading model...')

    decoder = torch.load(args.load)
    decoder.eval()
    try:
        ep_start = int(args.load.split('ep')[1].split('-')[0])
        iter_start = int(args.load.split('it')[1].split('.')[0]) + 1
    except: pass

#all_losses = []
loss_avg = 0

def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].view(1))

    loss.backward()
    decoder_optimizer.step()

    return loss.data.item() / args.chunk_len

base_name = os.path.splitext(os.path.basename(args.filename))[0]
def save(ep=0,its=0):
    base = 'save/' + base_name
    if ep == 0 and its == 0:
        save_filename = base + '.pt'
    else:
        save_filename = base + '-ep' + str(ep) + '-it' + str(its)  + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

if args.log_every != 0:
    log_filename = 'logs/' + base_name + '-log-' + str(round(time.time())) + '.csv'
    with open(log_filename, 'w') as lg:
        lg.write('steps,loss,smooth_loss,time_elapsed\n')

if args.log_sample != 0:
    sample_filename = 'logs/' + base_name + '-sample-' + str(round(time.time())) + '.txt'

def args_condition(given_args, count):
    if given_args != 0:
        if count % given_args == 0:
            return True
        else: return False
    else: return False

start = time.time()
smooth_loss = 0
smooth_time_iter = 0
try:
    #print("Training for %d epochs with %d iters" (% args.n_epochs, iters))
    for epoch in range(ep_start, args.n_epochs):
        for its in range(iter_start, iters + 1):
            its_time = time.time()
            if (epoch != 1 and epoch % args.epoch_random == 0) or args.all_rand:
                loss = train(*training_set(get_chunk(its,args.chunk_len,True)))
            else:
                loss = train(*training_set(get_chunk(its,args.chunk_len,False)))

            steps = ((epoch - 1) * iters) + its 
            time_iter = time.time() - its_time
            loss_avg += loss
            if steps == 1:
                loss_avg *= args.log_every

            if smooth_loss != 0: smooth_loss = smooth_loss * 0.999 + loss * 0.001
            else: smooth_loss = loss
            if smooth_time_iter != 0: smooth_time_iter = smooth_time_iter * 0.999 + time_iter * 0.001
            else: smooth_time_iter = time_iter

            if args_condition(args.print_every,steps):
                est_time =  ((args.n_epochs * iters) - steps) * smooth_time_iter / 60 / 60
                print('epoch %d/%d, iter %d/%d, loss: %.6f, time/iter: %.4f, time elapsed: %s, est: %.2fh'
                    % (epoch, args.n_epochs, its, iters, smooth_loss, smooth_time_iter, time_since(start), est_time))
                #print('epoch %d/%d, iter %d/%d, loss: %.6f, time/iter: %.4f, time elapsed: %s'
                #    % (epoch, args.n_epochs, its, iters, smooth_loss, smooth_time_iter, time_since(start)))
            if args_condition(args.log_every,steps) or steps == 1:
                with open(log_filename, 'a') as lg:
                    lg.write('%d,%.6f,%.6f,%.1f\n' % (steps, loss_avg / args.log_every, smooth_loss, time.time() - start))
                    #lg.write(str(steps) + ',' + str(loss_avg / args.log_every) + ',' + str(smooth_loss) + ',' + 
                    #    str(time.time() - start) + '\n')
                loss_avg = 0
            if args_condition(args.save_every,steps): save(epoch,its)
            if args_condition(args.sample_every,steps):
                try:
                    sample = generate(decoder, random.choice(string.ascii_letters), args.chunk_len*8)
                    print(sample, '\n')
                except RuntimeError:
                    sample = ''
                    pass
            if args_condition(args.log_sample,steps) or steps == 1:
                try:
                    sample = generate(decoder, random.choice(string.ascii_letters), args.chunk_len*8)
                    with open(sample_filename, 'a') as sp:
                        sp.write('=====\nsteps ' + str(steps) + ', loss: ' + str(smooth_loss) +
                            '\n=====\n' + sample + '\n\n')
                except RuntimeError: pass
    save()

except KeyboardInterrupt:
    save()

except:
    save()
    raise
