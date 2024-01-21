"""Evaluating concepts library

Library for evaluating concepts and storing summaries.

Evaluator: Evaluating concepts.
  create_concept_summary: Visualize and store the summary.
  alignment: Save images with or without the concept.
  concepts_accuracy: Extract probabilities of concepts from the target model.
  compare_predict: Compare original probabilities and concept probabilities.
  calculating_concept_score: Convert probabilities to importance scores.
  save_result: Sort by important concepts and save the summary.
  run: Run class function.
"""
import numpy as np
import os
import cv2
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.segmentation import mark_boundaries
from PIL import Image
from tools import target_model as model

class Evaluator:
    """Evaluating concepts
    
    Evaluate concepts via target models.
    """
    __slots__ = ('concept_dic', 
                 'images', 
                 'prob', 
                 'arg_model', 
                 'layer_num', 
                 'class_name', 
                 'alpha', 
                 'beta')
                
    def __init__(self, 
                 concepts, 
                 images, 
                 prob, 
                 arg_model,
                 layer_num, 
                 class_name, 
                 alpha, 
                 beta):
        """Define evaluating class
    
        Args:
          concepts: Dictionary-type concept.
          images: Original images.
          prob: Original probabilities.
          arg_model: Target model name.
          layer_num: Specific layer number.
          class_name: Class name (kit_fox).
          alpha: Sr (removed the concept) weight.
          beta: Se (removed all except the concept) weight.
        """
        self.concept_dic = concepts
        self.images = images
        self.prob = prob
        self.arg_model = arg_model
        self.layer_num = layer_num
        self.class_name = class_name
        self.alpha = alpha
        self.beta = beta

    def create_concept_summary(self, 
                               rank_patches, 
                               rank_masks, 
                               rank_image_numbers, 
                               concept_rank, 
                               image_origin, 
                               arg_model, 
                               layer_num,
                               class_name,
                               score):
        """Visualize and store the summary.
        
        Each concept is visualized along with the original image with boundaries.
        In addition, the number and importance score of each concept is visualized.
        The results are saved in the folder "./results/target model name/class name/".
        
        Args:
          rank_patches: Concept patches.
          rank_masks: Masks for object parts.
          rank_image_numbers: Original image number corresponding to the object.
          concept_rank: Concept importance ranking.
          image_origin: Original images.
          arg_model: Target model name.
          layer_num: Specific layer number.
          class_name: class name (kit_fox).
          score: Concept importance score.
        """
        num = 10
        concept_size = len(rank_patches)
        average_image_value=117
        plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * concept_size
        plt.rcParams['axes.facecolor'] = 'w'
        fig = plt.figure(figsize=(num * 2, 4 * concept_size))
        outer = gridspec.GridSpec(concept_size, 1, wspace=0., hspace=0.3)
        for n in range(concept_size):
            inner = gridspec.GridSpecFromSubplotSpec(
                2, num, subplot_spec=outer[n], wspace=0, hspace=0.3)
            concept_images = rank_patches[n]
            concept_patches = rank_masks[n]
            concept_image_numbers = rank_image_numbers[n]
            for i in range(len(concept_images)):
                ax = plt.Subplot(fig, inner[i])
                ax.imshow(concept_images[i])
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title('%10s Concept_%s score: %.4f'%(' ',concept_rank[n], score[n]))
                ax.grid(False)
                fig.add_subplot(ax)
                ax = plt.Subplot(fig, inner[i + num])
                mask = concept_patches[i]
                image = image_origin[concept_image_numbers[i]]
                ax.imshow(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(str(concept_image_numbers[i]))
                ax.grid(False)
                fig.add_subplot(ax)
        class_name = class_name.replace(' ','_')
        # saving path.
        save_path = os.path.join("./results", arg_model, class_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        summary_name = "{} layer{} concepts.png".format( class_name,str(layer_num))
        save_path = os.path.join(save_path, summary_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=3)
        plt.close(fig)

    def alignment(self, concept_dic, original_image):
        """Save images with or without the concept
        
        Dictionary-type concepts are grouped together by the same concept.
        To evaluate a concept, we sort it into a concept that exists in the same image.
        At this point, we divide the upsampled concept and the concept removed from the original image.
        
        Args:
          concept_dic: Dictionary-type concept.
          original_image: Original images.
        
        Returns:
          remove_h: Original images with concept removed.
          except_h: Upsampled concepts.
          image_idx: Index of original images.
          concept_idx: Index of concepts.
        """
        remove_concept = []
        except_concept = []
        concept_num = []
        image_num = []
        concepts_name = concept_dic['concepts']
        for num, concept in enumerate(concepts_name):
            iter_num = len(concept_dic[concept]['image_numbers'])
            for i in range(iter_num):
                mask = np.stack((concept_dic[concept]['masks'][i],)*3, axis=-1).astype(np.uint8)
                image = np.array(original_image[concept_dic[concept]['image_numbers'][i]])       
                # Making image with concept removed.
                image = image*np.invert(mask.astype('bool')).astype(np.uint8)
                # Add image with concept removed.
                remove_concept.append(image)
                # Add upsampled concept.
                except_concept.append(concept_dic[concept]['patches'][i])
                concept_num.append(num+1)
                image_num.append(concept_dic[concept]['image_numbers'][i])
        image_num = np.array(image_num)
        
        remove_h = []
        except_h = []
        image_idx = []
        concept_idx = []
        iter_num = np.max(image_num)+1
        for i in range(iter_num):
            idxs = np.where(image_num == i)[0]
            for idx in idxs:
                image_idx.append(i)
                remove_h.append(remove_concept[idx])
                concept_idx.append(concept_num[idx])
                except_h.append(except_concept[idx])
        return remove_h, except_h, image_idx, concept_idx
        
    def concepts_accuracy(self, remove_h, except_h):
        """Extract probabilities of concepts from the target model
        
        Getting probabilities of concepts by the target model.
        
        Args:
          remove_h: Original images with concept removed.
          except_h: Upsampled concepts.
          
        Return:
          concept_prob: Probability set of concepts.
        """
        evaluate_img = np.concatenate((remove_h, except_h))
        model_out = model.TargetModel(evaluate_img, self.arg_model, self.layer_num, self.class_name)
        result = model_out.run()
        # Use only probabilities.
        concept_prob = result[0]
        return concept_prob
        
    def compare_predict(self, remove_h, image_idx, concept_prob, original_prob):
        """Compare original probabilities and concept probabilities
        
        Compare the probabilities for the concept to the original probabilities.
        When the concept is removed, the probability is remove_prob, 
        when only the concept exists, the probability is except_prob.
        
        remove_prob returns the difference from the original probability.
        It is normalized against the differences of all concepts in the same image.
        
        except_prob is normalized through comparison with all concepts in the same image.
        
        Args:
          remove_h: Original images with concept removed.
          image_idx: Index of original images.
          concept_prob: Probability set of concepts.
          original_prob: Probability set of original images.
        
        Returns:
          score_r: Score for remove_h.
          score_e: Score for except_h.
        """
        split_idx = len(remove_h)
        image_num = np.array(image_idx)
        # Probabilities of remove_h.
        remove_prob = concept_prob[:split_idx]
        # Probabilities of except_h.
        except_prob = concept_prob[split_idx:]
        
        score_r = []
        score_e = []
        iter_num = np.max(image_num)+1
        for i in range(iter_num):
            idxs = np.where(image_num == i)[0]
            differ_prob = [original_prob[i] - remove_prob[idx] for idx in idxs]
            sum_prob = [except_prob[idx] for idx in idxs]
            for j in range(len(differ_prob)):
                score_r.append((differ_prob[j]/(np.sum(differ_prob)+1e-10))*100 )
            for j in idxs:           
                score_e.append((except_prob[j]/(np.sum(sum_prob)+1e-10))*100)
        score_r = np.array(score_r)
        score_e = np.array(score_e)
        return score_r, score_e
        
    
    def calculating_concept_score(self, image_idx, score_len, score_r, score_e, concept_idx):
        """Convert probabilities to importance scores
        
        Calculate the importance of a concept.
        We don't use probabilities directly, 
        we use the reverse order of magnitude of probabilities.
        If there are 10 concepts in one image, 
        it can have a value i/10 (i = 1, 2, 3, ..., 10).
        At this time, the score when the concept is removed has the weight of self.alpha, 
        and the score when there is only the concept has the weight of self.beta.
        We set self.beta to a higher value.
        
        Args:
          image_idx: Index of original images.
          score_len: Size of score.
          score_r: Score for remove_h.
          score_e: Score for except_h.
          concept_idx: Index of concepts.
          
        Returns:
          concept_score: Score for concept.
        """
        image_num = np.array(image_idx)
        rank_r = np.zeros(shape=(score_len))
        rank_e = np.zeros(shape=(score_len))
        num_img = len(np.bincount(image_num))
        for n in range(num_img):
            index = np.where(image_num == n)[0]
            if len(index) == 0:
                continue
            N_ic = float(len(index))
            sort_idx = np.argsort(score_r[index[0]:index[-1]+1])
            for i, idxs in enumerate(sort_idx):
                idx = index[idxs]
                concept_num = concept_idx[idx]
                if score_r[idx] < 0:
                    continue
                rank_r[concept_num-1] +=(i/N_ic) * self.alpha
            sort_idx = np.argsort(score_e[index[0]:index[-1]+1])
            for i, idxs in enumerate(sort_idx):
                idx = index[idxs]
                concept_num = concept_idx[idx]
                rank_e[concept_num-1] +=(i/N_ic) * self.beta
        concept_score = (rank_r + rank_e)/float(score_len)
        return concept_score
        
    def save_result(self, concept_score, concept_dic, original_image):
        """Sort by important concepts and save the summary
        
        Arrange the concepts in order of importance and save the summary in the form of an image.
        
        Args:
          concept_score: Score for concept.
          concept_dic: Dictionary-type concept.
        """
        idxs = np.argsort(concept_score)[::-1] +1
        # Arrange the concepts.
        concepts_name = [concept_dic["concepts"][idx-1] for idx in idxs]
        rank_patches = [concept_dic[concept]['patches'] for concept in concepts_name]
        rank_masks = [concept_dic[concept]['masks'] for concept in concepts_name]
        rank_image_numbers = [concept_dic[concept]['image_numbers'] for concept in concepts_name]
        # Making Summary.
        self.create_concept_summary(rank_patches,
                                    rank_masks,
                                    rank_image_numbers,
                                    idxs,
                                    original_image,
                                    self.arg_model,
                                    self.layer_num,
                                    self.class_name,
                                    np.sort(concept_score)[::-1])
    
    def run(self):
        """Run class function
        
        Execute all functions that exist in Evaluator class.
        Users can evaluate and summary by executing this function.
        """
        remove_h, except_h, image_idx, concept_idx = self.alignment(self.concept_dic, self.images)
        concept_prob = self.concepts_accuracy(remove_h, except_h)
        score_r, score_e = self.compare_predict(remove_h, image_idx, concept_prob, self.prob)
        score_len = len(self.concept_dic['concepts'])
        concept_score = self.calculating_concept_score(image_idx, score_len, score_r, score_e, concept_idx)
        self.save_result(concept_score, self.concept_dic, self.images)
