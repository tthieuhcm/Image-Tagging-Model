import matplotlib.pyplot as plt
import numpy as np

color = ['ob-', 'vg-', 'sr-', 'pc-', '*m-', 'hy-', 'xk-', 'Dw-']
color_bar = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
list_name = ['CE',
             'FL_1',
             'FL_2',  # Best validation loss',
             # 'Focal Loss, gamma=5, alpha=0.25, Final model',
             'FL_3',  # Best validation loss',
             # 'Focal Loss, gamma=2, alpha=1, multi-single dataset order, Final model',
             'BW_CE']  # , Best validation loss']
list_bar_name = ['C_SVC', 'nu_SVC']
list_threshold_name = ['SVM', 'NN 0.0', 'NN 0.1', 'NN 0.2', 'NN 0.3', 'NN 0.4', 'NN 0.5', 'NN 0.6', 'NN 0.7', 'NN 0.8',
                       'NN 0.9']


def F1_score_plot(list_F_score, list_threshold, number_of_plot):
    for index in range(number_of_plot):
        plt.plot(list_threshold, list_F_score[index], color[index], label=list_name[index])
    plt.xlabel("Threshold")
    plt.ylabel("Micro-average F1 score")
    plt.title('Micro-average F1 score Plots')
    plt.legend()
    plt.show()


def Precision_Recall_plot(list_P, list_R, number_of_plot):
    for index in range(number_of_plot):
        plt.plot(list_R[index], list_P[index], color[index], label=list_name[index])

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title('Precision-Recall Curve')
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
    plt.ylabel('Value')
    plt.xlabel('Measure')
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
    # Usually, we use 60 epochs, batch_size = 64, freeze to D2, SGD with lr=0.02 and decay to 1/10 each 30 epochs
    threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    precision_list_BCE = [.004031709393864304,
                          .5337933332356946,
                          .7086575335040718,
                          .7985501015716528,
                          .8524498782957688,
                          .8886106044147483,
                          .9164567289158243,
                          .9384831061374739,
                          .9555547094275577,
                          .9704967347148655,
                          .9718975740527971,
                          .9734759941390937,
                          .9749681708869706,
                          .9763472522269236,
                          .9779971791255281,
                          .9799289912683035,
                          .9818948055563546,
                          .9845424567188776,
                          .9893186003683216,
                          1]
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
    # precision_list_FocalLoss_gamma5_alpha025_last_model = [.004051942847378741,
    #                                                        .007441457870441845,
    #                                                        .024372003476320115,
    #                                                        .10513133601487253,
    #                                                        .3878001505758726,
    #                                                        .7584069812684117,
    #                                                        .9015892852932156,
    #                                                        .9294445864213295,
    #                                                        .9242608522589042,
    #                                                        .8943518091346412]
    precision_list_FocalLoss_gamma2_alpha1_DET_CLS_order_best_val = [0.004060288006601439,
                                                                     0.06863347140233593,
                                                                     0.2113823103883142,
                                                                     0.3978659653636652,
                                                                     0.5518156731465205,
                                                                     0.6356204563801614,
                                                                     0.6612591357170943,
                                                                     0.6451978933717958,
                                                                     0.5859458150901341,
                                                                     0.43855041398235695]
    # precision_list_FocalLoss_gamma2_alpha1_DET_CLS_order_last_model = [0.004057970616575151,
    #                                                                    0.07479946921046786,
    #                                                                    0.22292328103680364,
    #                                                                    0.4063167247780987,
    #                                                                    0.5502882973122214,
    #                                                                    0.6238871175229368,
    #                                                                    0.6434546062112318,
    #                                                                    0.6236271543965792,
    #                                                                    0.5613181466327545,
    #                                                                    0.41172656335861063]
    precision_list_WeightedBCELoss_best_val = [0.41169894075995755,
                                               0.7514492368239091,
                                               0.8302167698904215,
                                               0.8740860830313406,
                                               0.9008949066835195,
                                               0.9200883306880101,
                                               0.9345844504021414,
                                               0.9472231546753164,
                                               0.9580529650179005,
                                               0.9684630046785612]

    recall_list_BCE = [.9999999999999998,
                       .5768268243514592,
                       .5172973860540752,
                       .48137714375021007,
                       .4538053853656219,
                       .4288608276077636,
                       .4044563494968158,
                       .3787659674646524,
                       .347536307002637,
                       .29869767607486725,
                       .29118799717496796,
                       .2825882674126228,
                       .2726927421604372,
                       .26107212734009777,
                       .24691729263810108,
                       .22823607613342534,
                       .2023478627287672,
                       .16065687928923142,
                       .07970625997811984,
                       0]
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
    # recall_list_FocalLoss_gamma5_alpha025_last_model = [.9982017128240228,
    #                                                     .9963816641970715,
    #                                                     .9844148444748663,
    #                                                     .9114447427301914,
    #                                                     .7184008894498506,
    #                                                     .48587582916073985,
    #                                                     .30256725772086385,
    #                                                     .17267909179573512,
    #                                                     .08268560044789021,
    #                                                     .02684571724753254]
    recall_list_FocalLoss_gamma2_alpha1_DET_CLS_order_best_val = [0.9933627574528021,
                                                                  0.9421402584864711,
                                                                  0.8648040183508379,
                                                                  0.7733703146112315,
                                                                  0.6756792045596173,
                                                                  0.5778931354492056,
                                                                  0.4823841054362082,
                                                                  0.3858364607176134,
                                                                  0.2777255722767027,
                                                                  0.1473289797242626]
    # recall_list_FocalLoss_gamma2_alpha1_DET_CLS_order_last_model = [.9941244082369068,
    #                                                                 .9418514246826288,
    #                                                                 .8652451823115012,
    #                                                                 .7743851895521294,
    #                                                                 .6781896846765753,
    #                                                                 .5814204688207864,
    #                                                                 .4876028970426187,
    #                                                                 .3906951992260836,
    #                                                                 .28117377309928554,
    #                                                                 .1482271341553886]
    recall_list_WeightedBCELoss_best_val = [0.9880293075573562,
                                            0.9720683843005024,
                                            0.9591001341543086,
                                            0.9458566956761001,
                                            0.9315125038698328,
                                            0.917271507688071,
                                            0.8993498675656122,
                                            0.8735853599807338,
                                            0.8375012899453035,
                                            0.7690137938151397]

    F1_list_BCE = [.008031039967645448,
                   .554476363748944,
                   .5980426911479026,
                   .6006650275020956,
                   .5922982379584161,
                   .5785177119421688,
                   .5612280612233187,
                   .5397088048752349,
                   .5096957167853151,
                   .45680175836003173]
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
    # F1_list_FocalLoss_gamma5_alpha025_last_model = [.008071123047998202,
    #                                                 .01477258694936883,
    #                                                 .04756636559652132,
    #                                                 .18851791912282623,
    #                                                 .5036986280013434,
    #                                                 .5922956063475251,
    #                                                 .45308294706635677,
    #                                                 .29124797917920353,
    #                                                 .1517917131074491,
    #                                                 .052126748276309844]
    F1_list_FocalLoss_gamma2_alpha1_DET_CLS_order_best_val = [0.008087518948447904,
                                                              0.12794625457576275,
                                                              0.33972606143265793,
                                                              0.5254238312814608,
                                                              0.6074980545117292,
                                                              0.6053837401346,
                                                              0.5578328715427795,
                                                              0.48289539651033087,
                                                              0.37683808721820156,
                                                              0.22056138424625923]
    # F1_list_FocalLoss_gamma2_alpha1_DET_CLS_order_last_model = [0.008082947010294859,
    #                                                             0.13859228779050667,
    #                                                             0.3545099888870845,
    #                                                             0.5329806788486949,
    #                                                             0.6075808476450322,
    #                                                             0.6019056785856738,
    #                                                             0.5547911210060864,
    #                                                             0.4804155886647957,
    #                                                             0.3746693291237602,
    #                                                             0.21797891076473933]
    F1_list_WeightedBCELoss_best_val = [0.5812137031900262,
                                        0.8476386159653344,
                                        0.8900167583893277,
                                        0.9085562291955603,
                                        0.9159479113317123,
                                        0.9186777599279481,
                                        0.9166286855721448,
                                        0.9089152141514314,
                                        0.8937302693871211,
                                        0.8572907678540206]

    precision_list = list()
    precision_list.append(precision_list_BCE)
    precision_list.append(precision_list_FocalLoss_gamma2_alpha1)
    precision_list.append(precision_list_FocalLoss_gamma5_alpha025_best_val)
    # precision_list.append(precision_list_FocalLoss_gamma5_alpha025_last_model)
    precision_list.append(precision_list_FocalLoss_gamma2_alpha1_DET_CLS_order_best_val)
    # precision_list.append(precision_list_FocalLoss_gamma2_alpha1_DET_CLS_order_last_model)
    precision_list.append(precision_list_WeightedBCELoss_best_val)

    recall_list = list()
    recall_list.append(recall_list_BCE)
    recall_list.append(recall_list_FocalLoss_gamma2_alpha1)
    recall_list.append(recall_list_FocalLoss_gamma5_alpha025_best_val)
    # recall_list.append(recall_list_FocalLoss_gamma5_alpha025_last_model)
    recall_list.append(recall_list_FocalLoss_gamma2_alpha1_DET_CLS_order_best_val)
    # recall_list.append(recall_list_FocalLoss_gamma2_alpha1_DET_CLS_order_last_model)
    recall_list.append(recall_list_WeightedBCELoss_best_val)

    F1_list = list()
    F1_list.append(F1_list_BCE)
    F1_list.append(F1_list_FocalLoss_gamma2_alpha1)
    F1_list.append(F1_list_FocalLoss_gamma5_alpha025_best_val)
    # F1_list.append(F1_list_FocalLoss_gamma5_alpha025_last_model)
    F1_list.append(F1_list_FocalLoss_gamma2_alpha1_DET_CLS_order_best_val)
    # F1_list.append(F1_list_FocalLoss_gamma2_alpha1_DET_CLS_order_last_model)
    F1_list.append(F1_list_WeightedBCELoss_best_val)

    F1_score_plot(F1_list, threshold_list, 5)
    Precision_Recall_plot(precision_list, recall_list, 5)

    # Accuracy_C_SVC_elephant = 27.78
    # Precision_C_SVC_elephant = 10.85
    # Recall_C_SVC_elephant = 96.00
    # F1_C_SVC_elephant = 19.49
    # Accuracy_nu_SVC_elephant = 29.69
    # Precision_nu_SVC_elephant = 10.75
    # Recall_nu_SVC_elephant = 92.00
    # F1_nu_SVC_elephant = 19.25
    # plt.rcdefaults()
    # objects = ('Accuracy', 'Precision', 'Recall', 'F1')
    # C_SVC_performance = [Accuracy_C_SVC_elephant, Precision_C_SVC_elephant, Recall_C_SVC_elephant, F1_C_SVC_elephant]
    # nu_SVC_performance = [Accuracy_nu_SVC_elephant, Precision_nu_SVC_elephant, Recall_nu_SVC_elephant,
    #                       F1_nu_SVC_elephant]
    # title = 'C_SVC and nu_SVC models for elephant tag with setting -c4 -b1'
    # SVM_performance_plot(objects, title, C_SVC_performance, nu_SVC_performance)

    # plt.rcdefaults()
    # objects = ('linear', 'polynomial', 'radial basis function', 'sigmoid')
    # C_SVC_performance = [70, 69.18, 69.15, 69.17]
    # nu_SVC_performance = [72.27, 67.01, 67.74, 66.71]
    # title = 'C_SVC and nu_SVC models for animal tag with setting -c4 -b1'
    # SVM_performance_plot(objects, title, C_SVC_performance, nu_SVC_performance)
    #
    # Accuracy = [0.7227]
    # precision_SVM = [0.6890]
    # recall_SVM = [0.5527]
    # F1_SVM = [0.6134]
    # precision_NN = [0.4145833630438807, 0.6418379311107656, 0.804373291682934, 0.8933855799373013,
    #                 0.937636430802294, 0.9625333570539016, 0.9756414668655081, 0.9833340383232478,
    #                 0.987891881028934, 0.991255556365219]
    # recall_NN = [0.9999999999999967, 0.9980048845928899, 0.9920539369130715, 0.980324034260944,
    #              0.9604072787313785, 0.9305493447077813, 0.877644387877951, 0.7996628942932793,
    #              0.676378521550684, 0.46792336004953233]
    # F1_NN = [0.5861561415828868, 0.7812424266427194, 0.8884097034545861, 0.9348378736933595,
    #          0.9488852636802922, 0.9462711626756303, 0.9240520082077148, 0.8820382455113104,
    #          0.8029811127644081, 0.6357433284604244]
    # title = 'Compare precision for \'animal\' tag of Neural Network and SVM models'
    # ylabel = 'Precision'
    # both_precision = precision_SVM+precision_NN
    # compare_SVM_and_FocalLoss_performance_plot(list_threshold_name, title, ylabel, both_precision)
    #
    # title = 'Compare recall for \'animal\' tag of Neural Network and SVM models'
    # ylabel = 'Recall'
    # both_recall = recall_SVM+recall_NN
    # compare_SVM_and_FocalLoss_performance_plot(list_threshold_name, title, ylabel, both_recall)
    #
    # title = 'Compare F1 for \'animal\' tag of Neural Network and SVM models'
    # ylabel = 'F1'
    # both_F1 = F1_SVM+F1_NN
    # compare_SVM_and_FocalLoss_performance_plot(list_threshold_name, title, ylabel, both_F1)

    # # Data to plot
    # labels = 'No class - 4.80%', 'One class - 92.59%', 'Two classes - 1.56%', 'Three classes - 0.62%', 'More than four classes - 0.43%'
    # # labels = 'No class - 7.16%', 'One class - 56.82%', 'Two classes - 21.11%', 'Three classes - 9.10%', 'More than four classes - 5.80%'
    #
    # sizes = [15845, 305840, 5163, 2056, 1427]
    # # sizes = [1441, 11433, 4248, 1831, 1168]
    #
    # colors = ['yellowgreen', 'lightskyblue', 'gold', 'lightcoral', 'magenta']
    # explode = (0.1, 0, 0, 0, 0)
    #
    # patches, texts = plt.pie(sizes, explode=explode, colors=colors, shadow=False, startangle=90)
    # plt.legend(patches, labels, loc="best")
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()

    network = ['ResNet-50', 'ResNet-152', 'DenseNet-121', 'DenseNet-169', 'ResNeXt-101', 'NASNet-A (4 @ 1056)', 'AmoebaNet-C']  #, 'AmoebaNet-B', 'EfficientNet-B1', 'EfficientNet-B7']
    top1 =    [24         , 19.87       , 25.02         , 23.8          , 19.1         , 26                   , 16.5         ]  #, 15.7         , 21.2             , 15.6]
    param =   [26         , 60          , 8             , 14            , 84           , 5.3                  , 155.3        ]  #, 557          , 7.8              , 66.0]

    for i in range(7):
        plt.plot(param[i], top1[i], 'bo')
        if network[i] != 'DenseNet-169' and network[i] != 'AmoebaNet-C':
            plt.text(param[i]+2, top1[i]+0.2, network[i], fontsize=12)
        elif network[i] == 'DenseNet-169':
            plt.text(param[i]+2, top1[i]-0.5, network[i], fontsize=12)
        else:
            plt.text(param[i]-30, top1[i]+0.2, network[i], fontsize=12)

    plt.xlabel('Number of Parameters (Millions)')
    plt.ylabel('Imagenet Top 1 Error (%)')
    plt.axis([0, 170, 16, 28])
    plt.show()
