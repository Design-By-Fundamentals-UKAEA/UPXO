import numpy as np
from termcolor import colored

class artwork():
    """
    TO BE DEPRECATED
    """
    __slots__ = ('GrColArr',
                 )
    def __init__(self):
        pass

    def s_partitioned_tranition_probabilities(self,
                                              S,
                                              s_boltz_prob):
        fig = plt.figure(0,
                         figsize=(3.5, 3.5),
                         dpi=75,
                         )
        plt.scatter(np.arange(S), s_boltz_prob)
        plt.axis('auto')  # square, equal
        # plt.title("Q={}, m={}".format(Q, m+1), fontdict=font)
        plt.xlim([0., S])
        plt.xticks(np.linspace(0, S, 5))
        plt.ylim([0., 1.])
        plt.yticks(np.linspace(0, 1., 5))
        #plt.xlabel('Allowed no. of unique orientations', fontdict=font)
        #plt.ylabel('Probability of the unique orientation', fontdict=font)
        plt.grid(True)
        plt.show()

    def q_Col_Mat(self,
                  Q):
        """
        Summary line.

        State orientation based colour definitions: DESCRIPTION
        q_Col_Mat inputs
            1. Q        : No. of orientation states
        q_Col_Mat outputs
            1. GrColArr : Grain colour Array in RGB format. Q rows and 3 columns
        """
        if Q == 2:
            self.GrColArr = [[1, 0, 0],
                        [0, 0, 1]]
        elif Q == 3:
            self.GrColArr = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
        elif Q == 4:
            self.GrColArr = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
            self.GrColArr = np.vstack((self.GrColArr,
                                  [[1, 1, 0],
                                   [0, 1, 1],
                                   [1, 0, 1]][np.random.randint(3)]))
        elif Q > 4:
            gradient = 'random'
            if gradient == 'random':
                self.GrColArr = np.random.rand(Q, 3)
            elif gradient == 'GreyShades1':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                red = np.power(red, 2)
                green = np.arange(Q)/normFactor
                green = np.power(green, 2)
                blue = np.arange(Q)/normFactor
                blue = np.power(blue, 2)

                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'GreyShades2':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                green = np.arange(Q)/normFactor
                blue = np.arange(Q)/normFactor
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'RedShades':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                green = 0.*np.arange(Q)
                blue = 0.*np.arange(Q)
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'BlueShades':
                normFactor = Q-1
                red = 0.*np.arange(Q)
                green = 0.*np.arange(Q)
                blue = np.arange(Q)/normFactor
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'GreenShades':
                normFactor = Q-1
                red = 0.*np.arange(Q)
                green = np.arange(Q)/normFactor
                blue = 0.*np.arange(Q)
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'RedGreenShades':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                green = np.arange(Q)/normFactor
                blue = 0.*np.arange(Q)
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'Custom_01':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                red[::-1].sort()
                red = np.power(red, 3)
                green = np.arange(Q)/normFactor
                green[::-1].sort()
                green = np.power(green, 3)
                blue = np.arange(Q)/normFactor
                # blue[::-1].sort()
                blue = np.power(blue, 3)
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'lemon':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                # red[::-1].sort()
                red = np.power(red, 3)
                green = np.arange(Q)/normFactor
                # green[::-1].sort()
                green = np.power(green, 0.2)
                blue = np.arange(Q)/normFactor
                # blue[::-1].sort()
                blue = np.power(blue, 4)
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'Custom_02':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                # red[::-1].sort()
                red = np.power(red, 0.2)  # Fixed
                green = np.arange(Q)/normFactor
                green[::-1].sort()
                green = np.power(green, 0.3)  # Fixed
                blue = np.arange(Q)/normFactor
                blue[::-1].sort()
                blue = np.power(blue, 6)  # Fixed
                self.GrColArr = np.vstack((red, green, blue)).T
