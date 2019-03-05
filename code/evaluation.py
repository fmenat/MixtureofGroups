from astropy.table import Table
from sklearn.metrics import f1_score,accuracy_score
from scipy.stats import pearsonr
from tabulate import tabulate
import pandas as pd
import numpy as np
from .utils import *

class Evaluation_metrics(object):
    def __init__(self,class_infered,which='our1',N=None):
        self.which=which
        if self.which == 'our1':
            self.M = class_infered.M
            self.N = class_infered.N
            self.Kl = class_infered.Kl
            self.tested_model = class_infered.base_model
        elif self.which == 'keras':
            self.Kl = class_infered.output_shape[-1]
            self.N = N
            self.tested_model = class_infered
        elif self.which == 'raykar':
            self.T = class_infered.T
            self.Kl = class_infered.Kl
            self.N = class_infered.N
            self.tested_model = class_infered.base_model
        #and what about raykar or anothers

    def calculate_metrics(self,Z=[],Z_pred=[],y_o=[],yo_pred=[],conf_pred=[],conf_true=[],y_o_groups=[]):
        if len(yo_pred)!=0:
            self.T = yo_pred.shape[0]        
        to_return = []
        if self.which == 'our1' and len(conf_pred) == self.M: 
            to_return.append(self.report_results_wt_annot(conf_pred)) #intrisic metrics
            
        if len(Z) != 0: #if we have Ground Truth
            if self.which == 'our1' and len(y_o_groups) != 0 and len(conf_pred) == self.M:  #test set usually
                t_aux = self.report_results_groups(Z,y_o_groups) #groups performance
                if len(to_return) !=0: 
                    t_aux = pd.concat((to_return[-1],t_aux),axis=1) #append intrinsic metrics
                    to_return = [] #clean
                to_return.append(t_aux)
                
            t = self.report_results(Z_pred, Z, conf_pred, conf_true)
            if len(y_o) != 0 and len(yo_pred)!= 0: #if we have annotations and GT: maybe training set
                value_add = self.rmse_accuracies(Z, y_o, yo_pred)
                t["Average RMSE"] = np.mean(value_add)
            to_return.append(t)
            
        else: #if we dont have GT
            if len(y_o) != 0 and len(yo_pred) !=0: #If we have annotations but no GT: maybe trainig set
                to_return.append(self.report_results_wt_GT(y_o,yo_pred))
                    
        for table in to_return:
            print("A result\n",tabulate(table, headers='keys', tablefmt='psql'))
        return to_return
         
    def report_results(self,y_pred,y_true,conf_pred=[],conf_true=[]):
        """
            *Calculate metrics related to model and to confusion matrixs
            Needed: ground truth, for confusion matrix need annotations.
        """
        t = pd.DataFrame()#Table()
        t[""] = ["Global"]
        t["Accuracy"] = [accuracy_score(y_true,y_pred)]
        t["F1 (micro)"] = [f1_score(y_true=y_true, y_pred=y_pred, average='micro')]
        sampled_plot = 0
        if len(conf_true) != 0:
            KLs_founded = calculateKL_matrixs(conf_pred,conf_true)
            pearson_corr = []
            for m in range(len(conf_pred)):
                diagional_elements_pred = [conf_pred[m][f,f] for f in range(conf_pred[m].shape[0])]
                diagional_elements_true = [conf_true[m][f,f] for f in range(conf_true[m].shape[0])]
                #normalize diagonal
                diagional_elements_pred = (diagional_elements_pred-np.mean(diagional_elements_pred))/(np.std(diagional_elements_pred)+1e-10)
                diagional_elements_true = (diagional_elements_true-np.mean(diagional_elements_true))/(np.std(diagional_elements_true)+1e-10)
                pearson_corr.append(pearsonr(diagional_elements_pred, diagional_elements_true)[0])
                if np.random.rand() >0.5 and sampled_plot < 15:
                    compare_conf_mats(conf_pred[m], conf_true[m])
                    sampled_plot+=1
                    print("KL divergence: %.4f\tPearson Correlation between diagonals: %.4f"%(KLs_founded[m],pearson_corr[-1]))        
            #for now is mean.. maybe weighted 
            t["Average KL"] = np.mean(KLs_founded)
            t["Average PearsonCorr"] = np.mean(pearson_corr)
        return t

    def rmse_accuracies(self,Z_argmax,y_o,yo_pred): 
        """Calculate RMSE between accuracies of real annotators and predictive model of annotators
            Need annotations and ground truth
        """
        rmse_results = []
        for t in range(self.T):
            aux_annotations = np.asarray([(i,annotation) for i, annotation in enumerate(y_o[:,t]) if annotation != -1])
            t_annotations = aux_annotations[:,1] 
            
            gt_over_annotations = Z_argmax[aux_annotations[:,0]] #[Z_argmax[i] for i,annotation in aux_annotations]
            prob_data = yo_pred[t][aux_annotations[:,0]]

            acc_annot_real = accuracy_score(gt_over_annotations, t_annotations)
            if prob_data.shape[-1]>1: #if probabilities is handled
                acc_annot_pred = accuracy_score(t_annotations, prob_data.argmax(axis=-1))
            else: #if argmax is passed
                acc_annot_pred = accuracy_score(gt_over_annotations, prob_data) 

            rmse_results.append(np.sqrt(np.mean(np.square(acc_annot_real- acc_annot_pred ))))
        rmse_results = np.asarray(rmse_results)
        return rmse_results
    
    def report_results_wt_GT(self,y_o,yo_pred): #new
        """Calculate a comparison between annotators and predictive model of annotators without GT"""
        DT = pd.DataFrame()#Table()
        metric_acc = []
        metric_CE = []
        for t in range(self.T):
            aux_annotations = np.asarray([(i,annotation) for i, annotation in enumerate(y_o[:,t]) if annotation != -1])
            t_annotations = aux_annotations[:,1]
            
            prob_data = yo_pred[t][aux_annotations[:,0]]
            
            if prob_data.shape[-1]>1: #if probabilities is handled
                accuracy = accuracy_score(t_annotations, prob_data.argmax(axis=1))
                cross_entropy_loss = -np.mean(np.sum(keras.utils.to_categorical(t_annotations,num_classes=prob_data.shape[-1])*np.log(prob_data),axis=-1))
                metric_CE.append(cross_entropy_loss)
            else:
                accuracy = accuracy_score(t_annotations, prob_data)
            metric_acc.append(accuracy)
        DT["ACC imiting Annotator"] = [np.mean(metric_acc)]
        DT["CE imiting Annotator"] = [np.mean(metric_CE)]
        return DT
            
     
    def report_results_groups(self,Z_argmax,y_o_groups,added=True): #new
        """Calculate performance of the predictive model of the groups modeled"""
        t = pd.DataFrame()#Table()
        accs = []
        f1_s = []
        predictions_m = y_o_groups #by argmax
        for m in range(self.M):
            accs.append(accuracy_score(Z_argmax,predictions_m[m]))
            f1_s.append(f1_score(y_true=Z_argmax, y_pred=predictions_m[m], average='micro'))
        t["Accuracy"] = accs
        t["F1 (micro)"] = f1_s
        return t
        
    def report_results_wt_annot(self,conf_matrixs):
        """Calculate Intrinsic measure of only the confusion matrices infered """
        t = pd.DataFrame()#Table()
        identity_matrixs = np.asarray([np.identity(conf_matrixs.shape[1]) for m in range(len(conf_matrixs))])
        KLs_identity = calculateKL_matrixs(conf_matrixs,identity_matrixs)
        JSs_identity = calculateJS_matrixs(conf_matrixs,identity_matrixs)
        
        entropies = []
        mean_diagional = []
        for m in range(self.M):
            #compare_conf_mats(conf_matrixs[m])
            plot_confusion_matrix(conf_matrixs[m], np.arange(conf_matrixs[m].shape[0]),title="Group "+str(m),text=False)
            #New Instrisic measure
            entropies.append(Entropy_confmatrix(conf_matrixs[m]))
            mean_diagional.append(calculate_diagional_mean(conf_matrixs[m]))
            
        t["Groups"] = np.arange(len(conf_matrixs))        
        t["Entropy"] = entropies
        t["Diag Mean"] = mean_diagional
        t["KL to I"] = KLs_identity
        t["I similar % (JS)"] = 1-JSs_identity/np.log(2) #value betweeon [0,1]
        #t["Matrix-norm to identity"] = pendiente...
        inertia = distance_2_centroid(conf_matrixs)
        print("Inertia:",inertia)
        return t


"""
How to use it:

#Import it:
from evaluation import Evaluation_metrics
evaluate = Evaluation_metrics(gMixture,'our1')


#>>>>>>>>>>>>>>>>>>> Usuall train

#needed to evaluate other stuffs
aux = gMixture.calculate_extra_components(Xstd_train,y_obs,T=100,calculate_pred_annotator=True)
predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...

Z_train_pred = gMixture.base_model.predict_classes(Xstd_train)
#argmax groups
y_o_groups = predictions_m.argmax(axis=-1)

results = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt, y_o_groups=y_o_groups)


#>>>>>>>>>>>>>>>>>>> train bulk annotations without GT
results = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)


#>>>>>>>>>>>>>>>>>>> test or train without bulks annotation?--as repeats--- no annotations but ground truth
c_M = gMixture.get_confusionM()
Z_test_pred = gMixture.base_model.predict_classes(Xstd_test)
y_o_groups = gMixture.get_predictions_groups(Xstd_test) #obtain p(y^o|x,g=m)
#argmax groups
y_o_groups = y_o_groups.argmax(axis=-1)

results = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred,conf_pred=c_M, y_o_groups=y_o_groups)


#>>>>>>>>>>>>>>>>>>> test without GT
results = evaluate.calculate_metrics(conf_pred=c_M)
"""