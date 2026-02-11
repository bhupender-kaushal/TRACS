
                                            #############################                                   ######################
                                            #                                      Basic imports 
                                            #############################                                   ######################

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
import numpy as np
from math import *
import time
from util.visualizer import Visualizer
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
from sklearn.metrics import auc as sklearn_auc
import matplotlib.pyplot as plt

#### model flop and params calculation imports #####
from thop import profile, clever_format

##### shap library imports 
import shap
##### lime library imports
from lime import lime_image
from skimage.segmentation import mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm



                                            #############################                                   ######################
                                            #                   Basic functions for saving image and overlay CAM
                                            #############################                                   ######################

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)


def overlay_and_save_cam(image_tensor, cam_map, save_path, alpha=0.5):


    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().cpu().numpy()

    if image_tensor.ndim == 2:
        image_tensor = np.expand_dims(image_tensor, axis=0)
    if image_tensor.shape[0] == 1:
        image_tensor = np.repeat(image_tensor, 3, axis=0)


    img_np = np.transpose(image_tensor, (1, 2, 0))  
    img_np = np.clip(img_np, 0, 1)

    if cam_map.ndim == 3 and cam_map.shape[0] == 1:
        cam_map = cam_map[0]

    cam_map = np.clip(cam_map, 0, 1)
    cam_map = cv2.resize(cam_map, (img_np.shape[1], img_np.shape[0]))  # Resize if needed

    # finding heatmaps
    cam_uint8 = np.uint8(cam_map * 255)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_uint8 = np.uint8(img_np * 255)
    overlay = cv2.addWeighted(heatmap, alpha, img_uint8, 1 - alpha, 0)

    Image.fromarray(overlay).save(save_path)



                                            #############################                                   ######################
                                            #                   Main function for inference and explanation generation
                                            #############################                                   ######################

if __name__ == "__main__":
    ##defining function for shap explainer
    def shap_wrapper(img):
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)  # add batch dimension
        img = torch.from_numpy(img).permute(0,3,1,2).float()

        val_data['A']=img
        #N,1,512,512
        tricss.feed_data(val_data)
        tricss.test()
        visuals = tricss.get_current_segment()
        #N,1,512,512
        score = visuals['test_V'].numpy().mean(axis=(1, 2, 3)) 
        return score.reshape(-1, 1) 
#### until here #####

                                    ###############################                                   ######################

##defining function for lime explainer

    def lime_wrapper(img_np_batch):
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        img_tensor = torch.from_numpy(img_np_batch[:, :, :, 0:1]).permute(0, 3, 1, 2).float()

        # 2. Model Inference
        val_data['A'] = img_tensor.to(device) 
        tricss.feed_data(val_data)
        tricss.test()
        visuals = tricss.get_current_segment()
        
        obj_conf = visuals['test_V'].cpu().numpy().mean(axis=(1, 2, 3))
        
        probs = np.stack([1 - obj_conf, obj_conf], axis=1)
        return probs
                                                #### until here #####
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or test(inference)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    batchSize = opt['datasets']['train']['batch_size']

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if opt['phase'] == 'train':
            train_set = Data.create_dataset_xcad(dataset_opt, 'train')
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            training_iters = int(ceil(train_set.data_len / float(batchSize)))
            val_set = Data.create_dataset_xcad(dataset_opt, 'val')
            val_loader = Data.create_dataloader(val_set, dataset_opt, 'val')
            valid_iters = int(ceil(val_set.data_len / float(batchSize)))
        if opt['phase'] == 'test':
            val_set = Data.create_dataset_xcad(dataset_opt, 'test')
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
            valid_iters = int(ceil(val_set.data_len))
    logger.info('Initial Dataset Finished')

    # model
    
    tricss = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = tricss.begin_step
    current_epoch = tricss.begin_epoch
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    val_dice = 0
    if opt['phase'] == 'train':
        while current_epoch < n_epoch:
            current_epoch += 1
            for istep, train_data in enumerate(train_loader):
                iter_start_time = time.time()
                current_step += 1
                if current_epoch == 1 and istep == 0:
                    tricss.data_dependent_initalize(train_data, opt)
                    continue
                tricss.feed_data(train_data)
                tricss.optimize_parameters(current_epoch)
                # log
                if (istep+1) % opt['train']['print_freq'] == 0:
                    logs = tricss.get_current_log()
                    t = (time.time() - iter_start_time) / batchSize
                    visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, t, 'Train')
                    visuals = tricss.get_current_visuals()
                    visualizer.display_current_results_step(visuals)

                # validation
                if (current_step+1) % opt['train']['val_freq'] == 0:
                    tricss.test()
                    visuals = tricss.get_current_visuals(isTrain=False)
                    visualizer.display_current_results_step(visuals)

            if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
                logger.info('Saving models and training states.')
                tricss.save_network(current_epoch, current_step)
                visualizer.display_current_results(visuals, current_epoch, True)


            dice_per_case_score = 0
            for idata, val_data in enumerate(val_loader):
                tricss.feed_data(val_data)
                tricss.test()
                visuals = tricss.get_current_segment()
                predseg = visuals['test_V'].squeeze().numpy()
                predseg = (predseg + 1) / 2.
                predseg = (predseg > 0.5).astype(bool)

                label = val_data['F'].cpu().squeeze().numpy()
                label = (label + 1) / 2.
                label = (label > 0.5).astype(bool)
                dice = (visualizer.calculate_score(label, predseg, "dice"))
                dice_per_case_score += dice
            dice_case = (dice_per_case_score) / valid_iters
            if dice_case >= val_dice:
                val_dice = dice_case
                tricss.save_network(current_epoch, current_step, seg_save=True, dice=round(val_dice, 4))



        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')

          # Parameters with names


        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        file = open(result_path + 'test_score.txt', 'w')
        dice_per_case_score = []
        prec_per_case_score = []
        jacc_per_case_score = []
        acc_per_case_score = []
        recall_per_case_score = []
        sp_per_case_score = []
        auc_per_case_score = []

        ##for roc and pr curves
        labels_list = []
        preds_list = []

        total_cm = np.zeros((2, 2)) ## for confusion matrix.


        ## calculating inference speed
        inf_time=[]

        for idata,  val_data in enumerate(val_loader):

            dataInfo = val_data['P'][0].split('\\')[-1][:-4]
            
            time1 = time.time()
            tricss.feed_data(val_data)
            tricss.test()
            time2=time.time()
            inf_time.append( time2- time1)
            
            visuals = tricss.get_current_segment()

            

            #### Preparing images for shap and lime explainers ####
            img_for_shap = val_data['A'].clone().permute(0, 2, 3, 1)
            img_for_lime = val_data['A'].clone().squeeze(0).permute(1, 2, 0).cpu().numpy().astype('double')



            predseg = visuals['test_V'].squeeze().numpy()
            predseg = (predseg + 1) / 2.
            predseg = (predseg > 0.5).astype(bool)

            label = val_data['F'].cpu().squeeze().numpy()
            label = (label + 1) / 2.
            label = (label > 0.5).astype(bool)
    
    ## for roc and pr curves
            labels_list.append(label)
            preds_list.append(predseg)

    ## for CM
            total_cm += confusion_matrix(label.flatten(), predseg.flatten(), labels=[0, 1])



            data = val_data['A'].cpu().squeeze().numpy()
            data = (data+1)/2.
                                        # saving images (image label prediction and cam)to the results folder ## 
            savePath = os.path.join(result_path, '%d_data.png' % idata)
            save_image(data * 255, savePath)
            # savePath = os.path.join(result_path, '%d_pred.png' % (idata))
            # save_image(predseg * 255, savePath)
            # savePath = os.path.join(result_path, '%d_label.png' % (idata))
            # save_image(label * 255, savePath)



######defining profiler to calculate the input size flops and params of the segmentation network######
##########                              the break point added to calculate the flops and params  ##########################
            # flops, params = tricss.netG.findprofile(val_data, opt)
            # logger.info(f'FLOPS: {flops}, Params: {params}')
            # file.write(' FLOPS=%f, Params=%f \n' % (flops, params))
            # exit(0)


            cam_img=tricss.get_cam(val_data) # get the cam_map of the image.
            cam_name=["AblationCAM", "ScoreCAM", "LayerCAM", "HiResCAM"]
            for i in range(len(cam_img)):
                savePath = os.path.join(result_path, '%d_%s_cam.png' % (idata,cam_name[i]))
                overlay_and_save_cam(data, cam_img[i], savePath, alpha=1)



# ############ SHAP explainer code ##################       
#             print("Generating SHAP explanations...")      
#             img_for_shap = img_for_shap.cpu().numpy() 

#             height, width, channels = val_data['A'].shape[2], val_data['A'].shape[3], val_data['A'].shape[1]
#             masker = shap.maskers.Image("blur(16,16)", (height, width, channels)) 

        
#             explainer = shap.Explainer((shap_wrapper), masker)

#             shap_values = explainer(img_for_shap, max_evals=700)  

#             if torch.is_tensor(shap_values.values):
#                 shap_values.values = shap_values.values.cpu().numpy()
#             if torch.is_tensor(shap_values.data):
#                 shap_values.data = shap_values.data.cpu().numpy()

#             shap.image_plot(shap_values, show=False)
#             savePath = os.path.join(result_path, '%d_shap.png' % (idata))
#             plt.savefig(savePath, bbox_inches='tight', dpi=300)
#             plt.close()
# ############## ####### ########### saved shap images ########### ########### ##############

# ##################### lime processing #########################
#             print("Generating LIME explanations...") 
#             explainer_lime = lime_image.LimeImageExplainer()

#             img_for_lime = np.concatenate([img_for_lime] * 3, axis=-1) 
#             segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=0.2, sigma=1, convert2lab=False)

#             explanation = explainer_lime.explain_instance(img_for_lime, lime_wrapper, top_labels=1, num_samples=50,segmentation_fn=segmenter, random_seed=42)
#             temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10,hide_rest=False)
#             plt.imshow(mark_boundaries((temp + 1.0) / 2.0, mask))
#             savePathLime = os.path.join(result_path, '%d_lime.png' % (idata))
#             plt.savefig(savePathLime, bbox_inches='tight', dpi=300)
#             plt.close()


################ ########### saved lime images ########### ########### ##############

            dice = (visualizer.calculate_score(label, predseg, "dice"))
            prec = (visualizer.calculate_score(label, predseg, "prec"))
            jacc = (visualizer.calculate_score(label, predseg, "jacc"))
            acc = (visualizer.calculate_score(label, predseg, "accuracy"))
            recall = (visualizer.calculate_score(label, predseg, "recall"))
            sp = (visualizer.calculate_score(label, predseg, "sp"))
            auc= (visualizer.calculate_score(label, predseg, "auc"))

            file.write('%04d: process image... %03s | Dice=%f | Prec=%f | Jacc=%f \n' % (idata, dataInfo, dice, prec, jacc))

            print('%04d: process image... %s' % (idata, dataInfo))
            dice_per_case_score.append(dice)
            prec_per_case_score.append(prec)
            jacc_per_case_score.append(jacc)
            acc_per_case_score.append(acc)
            recall_per_case_score.append(recall)
            sp_per_case_score.append(sp)
            auc_per_case_score.append(auc)

# plotting ROC and PR curves

#         all_true = np.concatenate([img.flatten() for img in labels_list])
#         all_probs = np.concatenate([img.flatten() for img in preds_list])
#         fpr, tpr, _ = roc_curve(all_true, all_probs)
#         precision, recall, _ = precision_recall_curve(all_true, all_probs)
#         roc_auc = sklearn_auc(fpr, tpr)
#         pr_auc = sklearn_auc(recall, precision)
#         plt.figure()
#         plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.0])
#         plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate'), plt.title('Receiver Operating Characteristic')
#         plt.legend(loc="lower right"), plt.savefig(os.path.join(result_path, 'roc_curve.png'), dpi=300, bbox_inches='tight'), plt.close()
        
#         plt.figure()
#         plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
#         plt.xlim([0.0, 1.0]), plt.ylim([0.0, 1.0]),  plt.xlabel('Recall'), plt.ylabel('Precision'), plt.title('Precision-Recall Curve')
#         plt.legend(loc="lower left"), plt.savefig(os.path.join(result_path, 'pr_curve.png'), dpi=300, bbox_inches='tight'), plt.close()
       

# # plotting and saving confusion matrix        
#         normalized_cm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
#         disp = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=['Background', 'Vessel'])
#         fig, ax = plt.subplots(figsize=(10, 8))
#         disp.plot(ax=ax)
#         plt.savefig(os.path.join(result_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
#         plt.close()

## saving scores
        # print("score_dice_per_case: %.04f +- %.03f" % (np.mean(dice_per_case_score), np.std(dice_per_case_score)))
        # print("score_prec_per_case : %.04f +- %.03f" % (np.mean(prec_per_case_score), np.std(prec_per_case_score)))
        # print("score_jacc_per_case : %.04f +- %.03f" % (np.mean(jacc_per_case_score), np.std(jacc_per_case_score)))
        # print("score_acc_per_case : %.04f +- %.03f" % (np.mean(acc_per_case_score), np.std(acc_per_case_score)))
        # print("score_recall_per_case : %.04f +- %.03f" % (np.mean(recall_per_case_score), np.std(recall_per_case_score)))
        # print("score_sp_per_case : %.04f +- %.03f" % (np.mean(sp_per_case_score), np.std(sp_per_case_score)))
        # print("score_auc_per_case : %.04f +- %.03f" % (np.mean(auc_per_case_score), np.std(auc_per_case_score)))
        # print("Average Frame per second during inference: %.04f +- %.03f" % (1/np.mean(inf_time), np.std(1/np.array(inf_time))))

        # file.write('score_case | Dice=%f | Precision=%f | Jaccard=%f | Accuracy=%f | Recall=%f |Specificity=%f \n'#| AUC=%f \n'
        #            % (np.mean(dice_per_case_score), np.mean(prec_per_case_score), np.mean(jacc_per_case_score),np.mean(acc_per_case_score),np.mean(recall_per_case_score),np.mean(sp_per_case_score)))#,np.mean(auc_per_case_score)
        # file.write('S_std_case | Dice=%f | Precision=%f | Jaccard=%f | Accuracy=%f | Recall=%f |Specificity=%f  \n'#| AUC=%f \n'
        #            % (np.std(dice_per_case_score), np.std(prec_per_case_score), np.std(jacc_per_case_score),np.std(acc_per_case_score),np.std(recall_per_case_score),np.std(sp_per_case_score)))#,np.std(auc_per_case_score)
        # file.write('Average Frame per second during inference: %f +- %f \n'% (1/np.mean(inf_time), np.std(1/np.array(inf_time))))
        
        # total_params = sum(p.numel() for p in tricss.netG.parameters() if p.requires_grad) 
        # file.write('Total trainable parameters: %d \n' % total_params)

        # total_params = sum(p.numel() for p in tricss.netG.parameters()) 
        # file.write('Total parameters (trainable + non-trainable): %d \n' % total_params)




        # # =========================================================
        # # 2. PERFORM BOOTSTRAPPING (The "Robust" Part)
        # # =========================================================
        # print("Inference Done. Running Bootstrap Analysis...")
        # dice_scores = np.array(dice_per_case_score)

        # print(f"saving dice scores to {result_path} dice_scores.txt")
        # np.savetxt(result_path + 'dice_scores.txt', dice_scores)



        # # Configuration
        # n_bootstrap = 1000
        # bootstrap_means = []

        # for _ in range(n_bootstrap):
        #     # Resample with replacement (Simulate repeated random splits)
        #     # We pick N images randomly from the N available images
        #     indices = np.random.choice(len(dice_scores), len(dice_scores), replace=True)
        #     sample = dice_scores[indices]
        #     bootstrap_means.append(np.mean(sample))

        # bootstrap_means = np.array(bootstrap_means)

        # # Calculate Statistics
        # mean_score = np.mean(dice_scores)
        # std_score = np.std(dice_scores) # Standard Deviation of the raw data
        # ci_lower = np.percentile(bootstrap_means, 2.5)
        # ci_upper = np.percentile(bootstrap_means, 97.5)

        # # =========================================================
        # # 3. GENERATE ROBUSTNESS PLOT (Reviewers Love This)
        # # =========================================================
        # plt.figure(figsize=(8, 6))
        # plt.hist(bootstrap_means, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        # plt.axvline(ci_lower, color='red', linestyle='--', label='95% CI Lower')
        # plt.axvline(ci_upper, color='red', linestyle='--', label='95% CI Upper')
        # plt.axvline(mean_score, color='green', linewidth=2, label='Mean')
        # plt.title(f"Bootstrap Distribution of Dice Scores (TRACS)(N={n_bootstrap})")
        # plt.xlabel("Mean Dice Score")
        # plt.ylabel("Frequency")
        # plt.legend()
        # plt.savefig(result_path + 'bootstrap_robustness.png', dpi=500, bbox_inches='tight')
        # plt.close()
        # # =========================================================
        # # 4. WRITE FINAL REPORT
        # # =========================================================
        # report = f"""
        # ------------------------------------------------
        # ROBUSTNESS REPORT
        # ------------------------------------------------
        # Total Cases: {len(dice_scores)}
        # Mean Dice: {mean_score:.4f}
        # Std Dev:   {std_score:.4f}

        # Bootstrap Analysis (N={n_bootstrap} splits):
        # ------------------------------------------------
        # 95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]
        # Bootstrap Standard Error: {np.std(bootstrap_means):.4f}
        # ------------------------------------------------
        # """
        # from scipy import stats
        # dice_scores_baseline= np.loadtxt('/bhupender/Others/darl/MEDIA_CDARL/darl2/DARL/experiments/DARM_test_260127_173307/resultsdice_scores.txt')
        # stat, p_value = stats.wilcoxon(dice_scores, dice_scores_baseline)
        # print(f"P-Value vs Baseline: {p_value}")
        # print(report)
        # file.write(report)
        # file.write('\n')
        # file.write(f"P-Value vs Baseline: {p_value}\n")


# # =========================================================
#         # 5. GENERATE COMPARATIVE BOX PLOT (Clean & Professional)
#         # =========================================================
#         import seaborn as sns
#         import pandas as pd
#         import matplotlib.patches as mpatches

#         print("Generating Clean Comparative Box Plot...")

#         # 1. Prepare Data for Seaborn
#         df_box = pd.DataFrame({
#             'Score': np.concatenate([dice_scores_baseline, dice_scores]),
#             'Method': ['Baseline (DARL)'] * len(dice_scores_baseline) + ['TRACS (Ours)'] * len(dice_scores)
#         })

#         # Determine plot limits for clean bracket placement
#         y_min = df_box['Score'].min()
#         # Find the effective maximum (whiskers top) for bracket placement since fliers are hidden
#         # A rough robust estimate for the top whisker is Q3 + 1.5*IQR
#         q1 = df_box.groupby('Method')['Score'].quantile(0.25)
#         q3 = df_box.groupby('Method')['Score'].quantile(0.75)
#         iqr = q3 - q1

#         # Calculate theoretical top whisker for both and take the max
#         estimated_max_whisker = (q3 + 1.5 * iqr).max()

#         # Define figure with high DPI
#         fig, ax = plt.subplots(figsize=(6, 8), dpi=300)

#         # 2. Draw Box Plot (Appealing Style)
#         # Palette: Light gray vs. Vibrant Sky Blue
#         # linewidth=2 makes it bolder
#         # showfliers=False HIDES points outside whiskers
#         sns.boxplot(x='Method', y='Score', data=df_box,
#                     width=0.6, linewidth=2,
#                     palette=["#E0E0E0", "#4FC3F7"],
#                     showfliers=False, ax=ax)
#         # sns.stripplot(x='Method', y='Score', data=df_box, 
#         #               color='black', alpha=0.3, jitter=True, size=3)

#         # 3. Add Statistical Significance Bracket
#         # Calculate height for the bracket based on estimated max whisker
#         y_offset = 0.02  # Space above max whisker
#         y_line = estimated_max_whisker + y_offset
#         y_h = 0.015       # Height of bracket 'legs'

#         # Determine Star Rating
#         if p_value < 0.001:
#             sig_text = "***"
#         elif p_value < 0.01:
#             sig_text = "**"
#         elif p_value < 0.05:
#             sig_text = "*"
#         else:
#             sig_text = "ns"

#         # Draw the bracket lines (x-coords: 0=Baseline, 1=TRACS)
#         # Using somewhat thinner line for the bracket so it doesn't dominate
#         plt.plot([0, 0, 1, 1], [y_line, y_line + y_h, y_line + y_h, y_line], lw=1, c='k')

#         # Add P-value text above bracket
#         plt.text(0.5, y_line + y_h + 0.005, f"{sig_text}\n(p={p_value:.1e})",
#                  ha='center', va='bottom', color='k', fontsize=12)

#         # 4. Styling & Saving
#         ax.set_title("Segmentation Performance Comparison", fontsize=14, pad=20, fontweight='bold')
#         ax.set_ylabel("Dice Score", fontsize=12, fontweight='bold')
#         ax.set_xlabel("", fontsize=12) # No x-label needed
        
#         # Clean up axes
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.tick_params(axis='both', which='major', labelsize=11)

#         # Set Y-limit to accommodate bracket comfortably
#         ax.set_ylim(bottom=y_min - 0.05, top=y_line + 0.1)
        
#         # Subtler grid
#         ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
#         ax.set_axisbelow(True)

#         # Save
#         save_path_box = result_path + 'comparison_boxplot_clean.png'
#         plt.savefig(save_path_box, bbox_inches='tight', dpi=500)
#         plt.close()

#         print(f"Clean Comparative Box Plot saved to {save_path_box}")
        
        file.close()



        


