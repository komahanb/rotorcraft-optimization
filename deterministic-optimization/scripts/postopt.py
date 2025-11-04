# Import numpy
import numpy as np

# Import matplotlib
import matplotlib.pylab as plt

# Configure
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

# Optionally set font to Computer Modern to avoid common missing font errors
params = {
  'axes.labelsize': 20,
  'legend.fontsize': 14,
  'xtick.labelsize': 18,
  'ytick.labelsize': 18,
  'text.usetex': True}
plt.rcParams.update(params)

# Latex math
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath}']
plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'courier'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.color'] = 'r'

# Make sure everything is within the frame
plt.rcParams.update({'figure.autolayout': True})

# Set marker size
markerSize = 7.0

# bar chart settings
alpha = 0.9

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

mevery = 3

class PostOpt:
    """
    Class to handle plotting of post optimization data.
    """
    @staticmethod
    def plot_history(data, name):
        plt.figure()
        fig, ax = plt.subplots()

        # remove the upper and right borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.semilogy(data[:,0], data[:,2], '-', label='failure: collective',
                     ms=markerSize, mec='black', color=tableau20[0], markevery=mevery,
                     alpha=alpha)
        ax.semilogy(data[:,0], data[:,3], '-', label='failure: longitudinal',
                     ms=markerSize, mec='black', color=tableau20[2], markevery=mevery,
                     alpha=alpha)
        ax.semilogy(data[:,0], data[:,4], '-', label='failure: lateral',
                     ms=markerSize, mec='black', color=tableau20[4], markevery=mevery,
                     alpha=alpha)
        ax.set_ylabel('Infeasibility')
        ax.set_xlabel('Number of Optimizer Iterations')
        niters_labl = [1, 20, 40, 60, 80, 100]
        ax.set_yticks(np.logspace(-5, 0, 6, endpoint=True))
        ax.set_xticks([1,25,50,75])

        ax.legend(loc='center left', framealpha=0.0, bbox_to_anchor=(0.05, 0.5))

        # Make a second axis and plot the mass objective
        ax2 = ax.twinx()
        ax2.plot(data[:,0], data[:,1],
                 '-', label='mass',
                 ms=markerSize, mec='black', color=tableau20[6], markevery=mevery,
                 alpha=1.0)
        ax2.set_ylabel('Objective')
        ax2.tick_params('y', colors=tableau20[6])
        ax2.legend(loc='upper right', framealpha=0.0)
        ax2.set_yticks(np.linspace(0.5, 1, 6, endpoint=True))

        plt.savefig(name, bbox_inches='tight', pad_inches=0.05)

        return

if __name__ == "__main__":
    """
    """
    import pickle

    with open ('opt.hist', 'rb') as fp:
        hist = pickle.load(fp)

    inpFile = open("hist.log", "r")
    content = list(inpFile.readlines())
    inpFile.close()

    hist = []
    for line in content:
        entry = line.split(",")
        hist.append([int(entry[0]), float(entry[1]), float(entry[2]), float(entry[3]), float(entry[4])])

    print (hist)

    # convert to numpy array
    hist = np.array(hist)

    # Normalize
    hist[:,0] = hist[:,0]
    hist[:,1] = hist[:,1]/hist[0,1]
    hist[:,2] = hist[:,2]/hist[0,2]
    hist[:,3] = hist[:,3]/hist[0,3]
    hist[:,4] = hist[:,4]/hist[0,4]

    hist = np.where(hist < 0, 1.0e-5, hist)

    print (hist)

    PostOpt.plot_history(hist, "opthistory.pdf")
