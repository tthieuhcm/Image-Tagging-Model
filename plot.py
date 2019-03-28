import matplotlib.pyplot as plt
import numpy as np

color = ['ob-', 'og-', 'or-', 'oc-', 'om-', 'oy-', 'ok-', 'ow-']
color_bar = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
list_name = ['BCE Loss',
             'Focal Loss, gamma=2, alpha=1',
             'Focal Loss, gamma=5, alpha=0.25, Best validation loss',
             'Focal Loss, gamma=5, alpha=0.25, Final model',
             'SVM']
list_bar_name = ['C_SVC', 'nu_SVC']
list_threshold_name = ['SVM', 'NN 0.0', 'NN 0.1', 'NN 0.2', 'NN 0.3', 'NN 0.4', 'NN 0.5', 'NN 0.6', 'NN 0.7', 'NN 0.8', 'NN 0.9']


def F1_score_plot(list_F_score, list_threshold, number_of_plot):
    for index in range(number_of_plot):
        plt.plot(list_threshold, list_F_score[index], color[index], label=list_name[index])
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.title('F1 Score Plot')
    plt.legend()
    plt.show()


def Precision_Recall_plot(list_P, list_R, number_of_plot):
    for index in range(number_of_plot):
        plt.plot(list_R[index], list_P[index], color[index], label=list_name[index])
    plt.gca().invert_xaxis()

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title('Precision and Recall Plot')
    plt.legend()
    plt.show()


def SVM_performance_plot(objects, title, *args):
    y_pos = np.arange(len(objects))
    labels = []
    for i, arg in enumerate(args):
        plt.bar(y_pos - 0.15 + 0.3 * i, arg, width=0.3, color=color_bar[i], align='center', label=list_bar_name[i])
    for i in range(len(args[0])):
        labels.append(args[0][i])
        labels.append(args[1][i])
    plt.legend()
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy')
    plt.xlabel('Type of kernel')
    plt.title(title)

    for i, v in enumerate(labels):
        plt.text(y_pos[i // 2] + 0.3 * (i % 2 - 0.75), v + 1, str(v))

    plt.show()


def compare_SVM_and_FocalLoss_performance_plot(objects, title, ylabel, list_of_measure):
    y_pos = np.arange(len(objects))
    ps, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = plt.bar(y_pos, list_of_measure)
    ps.set_facecolor('r')
    p0.set_facecolor('b')
    p1.set_facecolor('b')
    p2.set_facecolor('b')
    p3.set_facecolor('b')
    p4.set_facecolor('b')
    p5.set_facecolor('b')
    p6.set_facecolor('b')
    p7.set_facecolor('b')
    p8.set_facecolor('b')
    p9.set_facecolor('b')
    plt.legend()
    plt.xticks(y_pos, objects)
    plt.ylabel(ylabel)
    plt.xlabel('Methods and Thresholds')
    plt.title(title)

    plt.show()


if __name__ == "__main__":
    # Usually, we use 60 epochs, batch_size = 64, freeze to D2, SGD with lr=0.002 and decay to 1/10 each 30 epochs
    threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    precision_list_BCE = [.004016778523489933, .0063070433181556034, .008352213464838245, .010243531128404131,
                          .012016890575180359, .013684898005091485, .015250969581037789, .016711000210461232,
                          .018045377484310485, .019189167188327267]
    precision_list_FocalLoss_gamma2_alpha1 = [0.004049893963631858,
                                              0.007578479812251471,
                                              0.2561843658491021,
                                              0.3295411785958549,
                                              0.3883874737234843,
                                              0.4340601456982287,
                                              0.46879702539463936,
                                              0.4946605818833678,
                                              0.5125942576601762,
                                              0.5229570329968404]
    precision_list_FocalLoss_gamma5_alpha025_best_val = [.004044016386757649,
                                                         .006982588574082241,
                                                         .022634433521081524,
                                                         .1016757831428169,
                                                         .38707034976066257,
                                                         .7733648481091132,
                                                         .9265998236144031,
                                                         .938669456942551,
                                                         .9658833522083777,
                                                         .9543639475185318]
    precision_list_FocalLoss_gamma5_alpha025_last_model = [.004051942847378741,
                                                           .007441457870441845,
                                                           .024372003476320115,
                                                           .10513133601487253,
                                                           .3878001505758726,
                                                           .7584069812684117,
                                                           .9015892852932156,
                                                           .9294445864213295,
                                                           .9242608522589042,
                                                           .8943518091346412]
    recall_list_BCE = [.9999999999999997,
                       .7884725703146754,
                       .6981063770537455,
                       .6436633249791145,
                       .6053350041771094,
                       .5755481295832173,
                       .5507276126824999,
                       .528847813979393,
                       .5083393669358581,
                       .4870832637148427]
    recall_list_FocalLoss_gamma2_alpha1 = [0.9974341270987434,
                                           0.9774699741434395,
                                           0.8762782379555313,
                                           0.826433634498626,
                                           0.7750090178133963,
                                           0.7246350505755904,
                                           0.6758430089360431,
                                           0.6281521956314876,
                                           0.5797685768604557,
                                           0.5292112858841381]
    recall_list_FocalLoss_gamma5_alpha025_best_val = [.9988070768238566,
                                                      .997111661961577,
                                                      .9845533264356126,
                                                      .9070172766137597,
                                                      .704837570551613,
                                                      .473821963634637,
                                                      .2930713518411176,
                                                      .22183919870380883,
                                                      .0674901727265713,
                                                      .023168032032855827]
    recall_list_FocalLoss_gamma5_alpha025_last_model = [.9982017128240228,
                                                        .9963816641970715,
                                                        .9844148444748663,
                                                        .9114447427301914,
                                                        .7184008894498506,
                                                        .48587582916073985,
                                                        .30256725772086385,
                                                        .17267909179573512,
                                                        .08268560044789021,
                                                        .02684571724753254]
    F1_list_BCE = [.00800141712571168,
                   .012513986444105805,
                   .016506936315155502,
                   .020166129909924786,
                   .023565958304098623,
                   .026734134307671235,
                   .02968002793810124,
                   .03239825184463508,
                   .03485350159551027,
                   .03692368618034306]
    F1_list_FocalLoss_gamma2_alpha1 = [0.00806703325285041,
                                       0.015040349407439844,
                                       0.3964612763794379,
                                       0.47119350843176905,
                                       0.5174569404416136,
                                       0.542912746363727,
                                       0.5535944623508913,
                                       0.5534709557945019,
                                       0.5441159912286695,
                                       0.5260655712623007]
    F1_list_FocalLoss_gamma5_alpha025_best_val = [.008055417624467041,
                                                  .013658061675463993,
                                                  .04425154414885294,
                                                  .18285382460524072,
                                                  .499715622593823526,
                                                  .5876220746679459,
                                                  .44530012410718406,
                                                  .3588660414592274,
                                                  .12616470752346487,
                                                  .04523787449899557]
    F1_list_FocalLoss_gamma5_alpha025_last_model = [.008071123047998202,
                                                    .01477258694936883,
                                                    .04756636559652132,
                                                    .18851791912282623,
                                                    .5036986280013434,
                                                    .5922956063475251,
                                                    .45308294706635677,
                                                    .29124797917920353,
                                                    .1517917131074491,
                                                    .052126748276309844]
    Accuracy = [0.7227]
    precision_SVM = [0.6890]
    recall_SVM = [0.5527]
    F1_SVM = [0.6134]
    precision_NN = [0.4145833630438807, 0.6418379311107656, 0.804373291682934, 0.8933855799373013,
                    0.937636430802294, 0.9625333570539016, 0.9756414668655081, 0.9833340383232478,
                    0.987891881028934, 0.991255556365219]
    recall_NN = [0.9999999999999967, 0.9980048845928899, 0.9920539369130715, 0.980324034260944,
                 0.9604072787313785, 0.9305493447077813, 0.877644387877951, 0.7996628942932793,
                 0.676378521550684, 0.46792336004953233]
    F1_NN = [0.5861561415828868, 0.7812424266427194, 0.8884097034545861, 0.9348378736933595,
             0.9488852636802922, 0.9462711626756303, 0.9240520082077148, 0.8820382455113104,
             0.8029811127644081, 0.6357433284604244]

    F1_list = list()
    F1_list.append(F1_list_BCE)
    F1_list.append(F1_list_FocalLoss_gamma2_alpha1)
    F1_list.append(F1_list_FocalLoss_gamma5_alpha025_best_val)
    F1_list.append(F1_list_FocalLoss_gamma5_alpha025_last_model)

    precision_list = list()
    precision_list.append(precision_list_BCE)
    precision_list.append(precision_list_FocalLoss_gamma2_alpha1)
    precision_list.append(precision_list_FocalLoss_gamma5_alpha025_best_val)
    precision_list.append(precision_list_FocalLoss_gamma5_alpha025_last_model)

    recall_list = list()
    recall_list.append(recall_list_BCE)
    recall_list.append(recall_list_FocalLoss_gamma2_alpha1)
    recall_list.append(recall_list_FocalLoss_gamma5_alpha025_best_val)
    recall_list.append(recall_list_FocalLoss_gamma5_alpha025_last_model)

    F1_score_plot(F1_list, threshold_list, 4)
    Precision_Recall_plot(precision_list, recall_list,4)

    plt.rcdefaults()
    objects = ('linear', 'polynomial', 'radial basis function', 'sigmoid')
    C_SVC_performance = [70, 69.18, 69.15, 69.17]
    nu_SVC_performance = [72.27, 67.01, 67.74, 66.71]
    title = 'C_SVC and nu_SVC models for animal tag with setting -c4 -b1'
    SVM_performance_plot(objects, title, C_SVC_performance, nu_SVC_performance)

    title = 'Compare precision for \'animal\' tag of Neural Network and SVM models'
    ylabel = 'Precision'
    both_precision = precision_SVM+precision_NN
    compare_SVM_and_FocalLoss_performance_plot(list_threshold_name, title, ylabel, both_precision)

    title = 'Compare recall for \'animal\' tag of Neural Network and SVM models'
    ylabel = 'Recall'
    both_recall = recall_SVM+recall_NN
    compare_SVM_and_FocalLoss_performance_plot(list_threshold_name, title, ylabel, both_recall)

    title = 'Compare F1 for \'animal\' tag of Neural Network and SVM models'
    ylabel = 'F1'
    both_F1 = F1_SVM+F1_NN
    compare_SVM_and_FocalLoss_performance_plot(list_threshold_name, title, ylabel, both_F1)
