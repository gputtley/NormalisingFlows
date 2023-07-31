import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import copy

hep.style.use("CMS")

def plot_histogram_with_ratio(hist_values1, hist_values2, bin_edges, name_1='Histogram 1', name_2='Histogram 2',xlabel="",name="fig", errors_1=0, errors_2=0):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # Compute bin centers

    # Plot the histograms on the top pad
    ax1.bar(bin_edges[:-1], hist_values1, width=np.diff(bin_edges), align='edge', alpha=1.0, label=name_1, color=(248/255,206/255,104/255), edgecolor=None)
    step_edges = np.append(bin_edges,2*bin_edges[-1]-bin_edges[-2])
    step_histvals = np.append(np.insert(hist_values1,0,0.0),0.0)
    ax1.step(step_edges, step_histvals, color='black')
    ax1.set_xlim([bin_edges[0],bin_edges[-1]])
    ax1.fill_between(bin_edges[:],np.append(hist_values1,hist_values1[-1])-np.append(errors_1,errors_1[-1]),np.append(hist_values1,hist_values1[-1])+np.append(errors_1,errors_1[-1]),color="gray",alpha=0.3,step='post',label="Uncertainty")

    # Plot the other histogram as markers with error bars
    ax1.errorbar(bin_centers, hist_values2, yerr=errors_2, fmt='o', label=name_2, color="black")

    # Get the current handles and labels of the legend
    handles, labels = ax1.get_legend_handles_labels()

    # Reverse the order of handles and labels
    handles = handles[::-1]
    labels = labels[::-1]

    # Create the reversed legend
    ax1.legend(handles, labels)

    #ax1.legend()
    ax1.set_ylabel('Density')
    hep.cms.text("Work in progress",ax=ax1)

    # Compute the ratio of the histograms
    ratio = np.divide(hist_values2,hist_values1)
    ratio_errors_1 = np.divide(errors_1,hist_values1)
    ratio_errors_2 = np.divide(errors_2,hist_values1)

    # Plot the ratio on the bottom pad
    ax2.errorbar(bin_centers, ratio, fmt='o', yerr=ratio_errors_2, label=name_2, color="black")

    ax2.axhline(y=1, color='black', linestyle='--')  # Add a horizontal line at ratio=1
    ax2.fill_between(bin_edges,1-np.append(ratio_errors_1,ratio_errors_1[-1]),1+np.append(ratio_errors_1,ratio_errors_1[-1]),color="gray",alpha=0.3,step='post')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Ratio')
    ax2.set_ylim([0.5,1.5])

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.1)

    # Show the plot
    print("Created "+name+".pdf")
    plt.savefig(name+".pdf")
    plt.close()


def plot_loss(train_loss_hist,test_loss_hist,xlabel="",max_val_before_nan=1000000):
    # Plot loss
    train_loss_hist =  [i if i < max_val_before_nan else np.nan for i in train_loss_hist]
    test_loss_hist =  [i if i < max_val_before_nan else np.nan for i in test_loss_hist]
    plt.figure(figsize=(10, 10))
    plt.plot(train_loss_hist, label='Train',color="blue")
    plt.plot(test_loss_hist, label='Test',color="red")
    plt.xlabel(xlabel)
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("plots/loss.pdf")
    plt.close()

