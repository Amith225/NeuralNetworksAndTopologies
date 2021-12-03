import matplotlib.pyplot as plt

for i in range(16):
    i = i + 1
    ax1 = plt.subplot(4, 4, i)
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
