import datetime
import gflags
import matplotlib.pyplot as plt
import os
import re
import sys

FLAGS = gflags.FLAGS

gflags.DEFINE_string('train_log_file', '/scr/r6/vigneshr/vigneshcaffe/projects/videovec_embedding/mednet_training_log_dir/caffe.bin.INFO', 'the training log file')
gflags.DEFINE_string('output_plot_dir', '/afs/cs.stanford.edu/u/vigneshr/www/misc/mednet_samp_skipg_tib2_v2/', 'output plot file to which all the plots will be saved')
gflags.DEFINE_integer('test_interval', 50, 'test is run per these many training iterations')
gflags.DEFINE_string('stats_key_words', 'Test net output #0: loss,Train net output #0: loss,Test net output #2: test_map',
  'comma separted list of keywords which corresponds to stats.'
  'For example if you want to read a line:'
  'Test net output #16: average_precision_per_class = 0.972'
  'then use keyword as Test net output #16: average_precision_per_class')


def parse_log_file(log_file, stats_key_words):
  stats_key_words_splits=stats_key_words.split(',')
  plot_values = {}
  for stats_key_word in stats_key_words_splits:
    plot_values[stats_key_word] = []
  # Open the log file
  log_file_data = open(log_file)
  if not log_file_data:
    print('Could not open log file: %s'%(log_file))
    sys.exit(1)
  else:
    # Read line by line (inefficient since we run it at every refresh)
    for line in log_file_data:
      for stats_key_words in stats_key_words_splits:
        match_idx = line.find(stats_key_words)
        if (match_idx >= 0):
          # Strip the time and convert it to datetime format
          dtime = datetime.datetime.strptime('%s 2015'%(line[1:14]), '%m%d %H:%M:%S %Y')
          # Get the reminder after removing the key words
          line_residue = line[(match_idx)+len(stats_key_words):]
          # In case the loss weight is provided, we will read that as well
          line_residue_splits = line_residue.split('(*')
          # First get the loss multiplied values
          true_value = re.findall('(-?\d+\.?\d*)', line_residue_splits[0])
          iter_value = -1
          # if the iter number is also provided in the  line, use it as well
          if len(true_value)==2 and 'iter' in line_residue_splits[0]:
            iter_value = int(true_value[0])
            plot_values[stats_key_words].append((dtime, iter_value, float(true_value[1])))
          elif len(true_value)==1:
            plot_values[stats_key_words].append((dtime, float(true_value[0])))
          else:
            print('Found more than values in the line: %s --> %d'%(line, len(true_value)))
          # if the loss weighted part is present, plot that as well
          if len(line_residue_splits) == 2:
            weighted_value = re.findall('(-?\d+\.?\d*)', line_residue_splits[1])
            if len(weighted_value) == 2:
              weighted_key = stats_key_words + ' with weight %s'%(weighted_value[0])
              if weighted_key not in plot_values:
                plot_values[weighted_key] = []
              if iter_value >= 0:
                plot_values[weighted_key].append((dtime, iter_value, float(weighted_value[1])))
              else:
                plot_values[weighted_key].append((dtime, float(weighted_value[1])))
            else:
              print('Did not find exactly 2 values in the loss weighted part in: %s --> %d'%(line, len(weighted_value)))
          break
  return plot_values

def main(argv):
  try:
    argv = FLAGS(argv)  # parse flags
  except gflags.FlagsError, e:
    print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
    sys.exit(1)

  # Parse the log file to get all the stats
  plot_values = parse_log_file(FLAGS.train_log_file, FLAGS.stats_key_words)
  idx = 0
  for key in plot_values:
    idx += 1;
    fid = open('%s/data_%03d.txt'%(FLAGS.output_plot_dir, idx), 'w')
    fid.write('# ---------------> %d:%s <----------------\n'%(idx, key))
    val_ctr = 0
    for value in plot_values[key]:
      if (len(value) == 3):
        fid.write('%d, %f\n'%(value[1], value[2]))
      else:
        fid.write('%d, %f\n'%(val_ctr*FLAGS.test_interval, value[1]))
        val_ctr = val_ctr + 1
    fid.close()
    #print(plot_values[key])
    if len(plot_values[key])>0:
      fig = plt.figure(figsize=(20,16))
      dtimes = [p[0] for p in plot_values[key]]
      if len(plot_values[key][0])==3:
        ax = fig.add_subplot(2,1,1)
        iters = [p[1] for p in plot_values[key]]
        values = [p[2] for p in plot_values[key]]
        ax.plot(iters, values)
        ax = fig.add_subplot(2,1,2)
        ax.plot(dtimes, values)
        plt.title(key)
      else:
        ax = fig.add_subplot(1,1,1)
        values = [p[1] for p in plot_values[key]]
        ax.plot([r*FLAGS.test_interval for r in range(len(values))], values)
        plt.title(key)
      plt.grid(True)
      fig.savefig('%s/plot_%03d.png'%(FLAGS.output_plot_dir, idx))
      plt.close()


if __name__ == "__main__":
  main(sys.argv)
