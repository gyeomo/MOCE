"""Discovering concepts library

Library for discovering concepts from a set of images by activation maps and gradients.

Clustering: Clustering for concept discovery.
  upsampling: Resize image while maintaining aspect ratio.
  data_preprocessing: Calculate relevance.
  act_up: Resize and binarize activation maps.
  act_separate: Separate components of masks.
  act_unique: Select the non-overlapping mask.
  cropping: Project the original image as a part using a mask.
  vector_extract: Convert parts to vector form using target model.
  make_cluster: Discovering concepts that exist in a set of images using clustering.
  run: Run class function.
"""
import numpy as np
import cv2
from scipy import ndimage
import numpy as np
import sklearn.cluster as cluster
from PIL import Image
from tools import target_model as model

class Clustering:
    """Clustering for discovering concepts
    
    Discover concepts from the set of images.
    """
    __slots__ = ('data', 
                 'images', 
                 'arg_model', 
                 'layer_num', 
                 'class_name', 
                 'gamma',
                 'img_size')
    def __init__(self, 
                 data, 
                 images, 
                 arg_model, 
                 layer_num, 
                 class_name, 
                 gamma):
        """Define clustering class
        
        Args:
          data: Activation maps, gradients.
          images: Original images.
          arg_model: Target model name.
          layer_num: Specific layer number.
          class_name: Class name (kit_fox).
          gamma: Threshold for binarization.
        """
        self.data = data
        self.images = images
        self.arg_model = arg_model
        self.layer_num = layer_num
        self.class_name = class_name
        self.gamma = 100-gamma
        self.img_size = images[0].shape[0]
        
    def upsampling(self, input_image, part, edges):
        """Resize image while maintaining aspect ratio
        
        Upsample a small image to the input image size while keeping the aspect ratio.
        
        Args:
          input_image: Original image.
          part: Mask corresponding to the object part.
          edges: x y coordinates of the masked part.
        Returns:
          Upsampled image patch.
        """
        img_size = input_image.shape[0]
        part_size = np.array(part).shape
        h1, h2, w1, w2 = edges[0].min(), edges[0].max(), edges[1].min(), edges[1].max()        
        I_c = np.array(part[h1:h2, w1:w2],dtype='uint8')
        (h_, w_) = I_c.shape[:2]
        h_ += 1e-5 
        w_ += 1e-5
        # Calculate aspect ratio.
        if w_>h_:
            ratio = img_size / w_
            dim = (img_size, int(h_ * ratio))
        else:
            ratio = img_size / h_
            dim = (int(w_ * ratio), img_size)
        try:
            I_r = cv2.resize(I_c, dim, interpolation=cv2.INTER_AREA)
            (h, w) = I_r.shape[:2]
            I_b = np.zeros((part_size[0], part_size[1], 3))
            startx = int(img_size/2 - (w)/2)
            starty = int(img_size/2 - (h)/2)
            I_b[starty:starty + h, startx:startx + w] = I_r        
            return np.array(I_b, dtype='uint8')
        except:
            return 0

    def data_preprocessing(self, data):
        """Calculate relevance
        
        Calculate relevance with activations and gradients.
        Relevance is considered the importance of activation maps.
        
        Args:
          data: Set of activations and gradients.
        
        Retuns:
          activations: Activation maps.
          relevance: Product of activations and gradients.
        """
        activations = np.concatenate(data[0], 0)
        gradients = np.concatenate(data[1], 0)
        gradients = np.mean(gradients, (2,3))
        relevance = np.mean(activations, (2,3)) * gradients
        return activations, relevance

    def act_up(self, activations, relevance):
        """Resize and binarize activation maps
        
        For computational efficiency, we will only use the top 50% activation maps by relevance.
        The activation map is smaller than the input image.
        Increase this back to the original image size, 
        and leave only the top self.gamma (10)% as 1, the rest as 0 for making mask.
        
        Args:
          activations: Activation maps.
          relevance: Product of activations and gradients.
          
        Returns:
          mask_b: Masks created by binarizing the activation maps.
        """
        mask_b = []
        idxs = len(activations)
        for idx in range(idxs):
            mask = []
            # Use only half 
            if len(relevance[idx]) > 1000:
                relevances = np.argsort(relevance[idx])[::-1][:500]
            else:
                relevances = np.argsort(relevance[idx])[::-1][:int(len(relevance[idx])/2)]
            for i in relevances: 
                mask.append(cv2.resize(activations[idx][i], (self.img_size,self.img_size)))
            mask_b.append(mask)
        for idx in range(idxs):
            iter_num = len(mask_b[idx])   
            for i in range(iter_num):
                # Calculating threshold point.
                thre = np.percentile(mask_b[idx][i], [0,25,50,int(self.gamma),100])[3]
                # making mask by binarization.
                _, mask_b[idx][i] = cv2.threshold(mask_b[idx][i], thre, 1, cv2.THRESH_BINARY)
        return mask_b
    
    def act_separate(self, mask_b):
        """Separate components of masks
        
        A mask may consist of several components.
        We use this function to use only the largest area.
        Separate components using scipy's ndimage.label().
        
        Args:
          mask_b: Masks created by binarizing the activation maps.
          
        Returns:
          mask_s: Masks consisting of the largest component.
        """
        mask_s = []
        mask_b_size = len(mask_b)
        for idx in range(mask_b_size):
            mask = []
            iter_num = len(mask_b[idx])
            for i in range(iter_num):
                # Separating component in the mask.
                label_im, nb_labels = ndimage.label(mask_b[idx][i])
                separates = []
                mask_max = []
                if nb_labels > 0:
                    for j in range(nb_labels):
                        mask_compare = np.full(np.shape(label_im), j+1) 
                        separate = np.equal(label_im, mask_compare).astype(int)
                        separates.append(separate)
                        mask_max.append(np.sum(separate))
                    if len(mask_max) > 0:
                        # Use only largest component. 
                        separates = np.array(separates[np.argmax(mask_max)])
                        mask.append(separates)
                    else:
                        # If there is only one element, just add it.
                        mask.append(mask_b[idx][i])
                else:
                    # Zero mask
                    mask.append(mask_b[idx][i])
            mask_s.append(mask)
        return mask_s

    def act_unique(self, mask_s):
        """Select the non-overlapping mask
        
        Multiple masks are created for one image.
        We want to exclude overlapping masks.
        Save the first mask and then save the mask 
        when jaccard score is less than 0.3.
        
        Args:
          mask_s: Masks consisting of the largest component.
          
        Returns:
          mask_u: Set of unique masks.
        """
        mask_u = []
        min_size = 0.005
        mask_s_size = len(mask_s)
        for idx in range(mask_s_size):
            masks = []
            iter_num = len(mask_s[idx])
            for i in range(iter_num):
                mask = mask_s[idx][i]
                # If mask is too small, it will remove.
                if np.mean(mask) > min_size:
                    flag = True
                    for seen_mask in masks:
                        # Calculating jaccard score.
                        jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                        if jaccard > 0.3:
                            flag = False
                            break
                    if flag:
                        # Adding mask in set of masks.
                        masks.append(mask)
            if len(masks) > 0:
                mask_u.append(masks)
        return mask_u

    def cropping(self, mask_u):
        """Project the original image as a part using a mask
        
        We get a part of the image by projecting a unique mask onto the input image.
        Since the part is smaller than the input image, 
        upsampling it while maintaining aspect ratio.
        
        Args:
          mask_u: Set of unique masks.
          
        Returns:
          image_u: Upsampled parts of the original image.
          image_num: Index of original image.
        """
        image_u = []
        image_num = []
        mask_u_size = len(mask_u)
        for idx in range(mask_u_size):
            image_parts = []
            input_image = np.array(self.images[idx])
            iter_num = len(mask_u[idx])
            for i in range(iter_num):
                stacked_img = np.stack((mask_u[idx][i],)*3, axis=-1)
                part = (input_image * stacked_img)
                edges = np.where(part > 0)
                # Exception handling.
                if len(edges[0]) < 1:
                    image_parts.append(np.uint8(np.zeros((self.img_size, self.img_size, 3))))
                    continue
                # Upsampling.
                ROI = self.upsampling(input_image, part, edges)
                if not isinstance(ROI,int):
                    if np.sum(ROI) > 1:
                        image_parts.append(np.uint8(cv2.resize(np.uint8(ROI), 
                                                        dsize=(self.img_size,self.img_size), 
                                                        interpolation=cv2.INTER_CUBIC)))
                # Exception handling.
                else:
                    image_parts.append(np.uint8(np.zeros((self.img_size, self.img_size, 3))))
            if len(image_parts) > 0:
                for i in range(len(image_parts)):
                    image_num.append(idx)
                image_u.append(np.array(image_parts))
        image_u = np.concatenate(image_u, 0)
        return image_u, image_num

    def vector_extract(self, image_u):
        """Convert parts to vector form using target model
        
        Put the part of the image as input to the target model.
        We only use activation maps here.
        The activation maps are converted into vectors.
        
        Args:
          image_u: Upsampled parts of the original image.
        
        Returns:
          concept_vectors: Vectors of part's activations.
        """
        model_out = model.TargetModel(image_u, self.arg_model, self.layer_num, self.class_name)
        result = model_out.run()
        # Activation maps -> vector.
        concept_vectors = np.mean(np.concatenate(result[1], 0),(2,3))
        return concept_vectors

    def make_cluster(self, mask_u, image_u, image_num, concept_vectors):
        """Discovering concepts that exist in a set of images using clustering
        
        We cluster the concept vectors with the K-Means clustering algorithm.
        A cluster becomes a concept.
        
        Args:
          mask_u: Set of unique masks.
          image_u: Upsampled parts of the original image.
          image_num: Index of original image.
          concept_vectors: Vectors of part's activations.
        
        Returns:
          concept_dic: Dictionary-type concept
        """
        # Cluster number is set to 25.
        km = cluster.KMeans(25, random_state=22)
        d = km.fit(concept_vectors)
        centers = km.cluster_centers_
        d = np.linalg.norm(np.expand_dims(concept_vectors, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
        asg, cost = np.argmin(d, -1), np.min(d, -1)
        
        concept_dic = {}
        concept_dic['label'] = asg
        concept_dic['cost'] = cost
        mask_u =np.concatenate(mask_u, 0)
        # If the number of images in each cluster is less than "min_imgs", that cluster is removed.
        min_imgs = 20
        max_imgs = -1
        concept_number, concept_dic['concepts'] = 0, []
        image_num = np.array(image_num)
        label_iter = concept_dic['label'].max() + 1
        for i in range(label_iter):
            label_idxs = np.where(concept_dic['label'] == i)[0]
            # Sort by cost before removing same image number.
            concept_costs = concept_dic['cost'][label_idxs]
            label_idxs = label_idxs[np.argsort(concept_costs)[:max_imgs]]
            unique_iter = np.unique(image_num[label_idxs])
            concept_idxs_copy = []
            # Remove the same image number.
            for j in unique_iter:
                concept_idxs_copy_ = label_idxs[np.where(image_num[label_idxs] == j)[0][0]]
                concept_idxs_copy.append(concept_idxs_copy_)
            label_idxs = np.array(concept_idxs_copy)
            # Sort by cost after removing same image number.
            concept_costs = concept_dic['cost'][label_idxs]
            concept_idxs = label_idxs[np.argsort(concept_costs)[:max_imgs]]
            if len(concept_idxs) < min_imgs:
                continue
            # Choose 10 images for each concept.
            concept_idxs = concept_idxs[:10]
            patches = [image_u[idx] for idx in concept_idxs]
            masks = [mask_u[idx] for idx in concept_idxs]
            image_num_ = [image_num[idx] for idx in concept_idxs]
            concept_number += 1
            concept = 'concept{}'.format( concept_number)
            # Save concept in dictionary.
            concept_dic['concepts'].append(concept)
            concept_dic[concept] = {
                'patches': patches,
                'masks': masks,
                'image_numbers': image_num_,
            }
        return concept_dic
    
    def run(self):
        """Run class function
        
        Execute all functions that exist in Clustering class.
        Users can discover concepts by executing this function.
        
        Returns:
          Clustering results
        """
        activations, relevance = self.data_preprocessing(self.data)
        mask_b = self.act_up(activations, relevance)
        mask_s = self.act_separate(mask_b)
        mask_u = self.act_unique(mask_s)
        image_u, image_num = self.cropping(mask_u)
        concept_vectors = self.vector_extract(image_u,)
        return self.make_cluster(mask_u, image_u, image_num, concept_vectors)
