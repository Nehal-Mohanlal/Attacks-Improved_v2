from utils.logs import logger_module 
import numpy as np 
logger = logger_module(logger_name=__name__) 


class FunctionalityPreservingAttack:

    def __init__(self):
        self.l2_distances = []
    
    @staticmethod
    def make_pred(sample, classifier): 
        single_pred = classifier.predict([sample])
        return single_pred 
    
    def feature_changes(initial_sample, final_sample,features):
        changes_dict = {i: {"from":int(old), "to":new} for i, (old,new) in enumerate(zip(initial_sample,final_sample)) if old!=new}
        
        changes = []
        for index, key in enumerate(changes_dict):
            change = f"Feature: {features[key]} changed: {changes_dict[key]}"
            changes.append(change)

        print(changes)
        
        
    def make_initial_adversarial(self, sampled_row, perturbation, classifier):
        lst = sampled_row.tolist().copy() #removed [0]
        logger.info(f"initial sample: {lst}")
    
        adv_indices = [i for i, x in enumerate(lst) if x == 0] # Track Indices
        logger.info(f"Identified features to perturb: {len(adv_indices)}") 

        adversarial_sample = list(map(lambda x: x+perturbation if x==0 else x, lst))
        logger.info(f"perturbed sample: {adversarial_sample}")
        prediction = FunctionalityPreservingAttack().make_pred(adversarial_sample, classifier) 
                
        if prediction == 0: 
            logger.info("Initial perturbation successful") 
            return adversarial_sample, adv_indices
        else: 
            logger.info("perturbation failed")
            return None, None

    def decrement_list(self, lst, perturbation, adversarial_indices, classifier):
        lst2 = lst.copy()
        final_adv_indices = [] # find all the features that have been perturbed.  
        logger.info(f"Identified indices to check if perturbation was necessary: {len(adversarial_indices)}")
        
        for j in adversarial_indices: 
            lst2[j] -= perturbation 
            check_pred = FunctionalityPreservingAttack().make_pred(lst2, classifier) 
            if check_pred != 0:
                lst2[j] += perturbation
                final_adv_indices.append(j)  
        
        logger.info(f"{len(adversarial_indices) - len(final_adv_indices)} features were identified as uneccessary to perturb: features to be optimized: {len(final_adv_indices)} features")        
        
        logger.info(f"Semi Optimized sample: {lst2}")
        return lst2, final_adv_indices ## Return final indices


    def optimize(self, sample, final_adversarial_sample,final_adv_indices, classifier): 
                    
        logger.info(f"final adversarial indices to check: {len(final_adv_indices)}")
        
        for j in final_adv_indices: ## trace all the features that removing the perturbation caused incorrect prediction
            while final_adversarial_sample[j]>sample[j]: 
                final_adversarial_sample[j] -= 1
                if FunctionalityPreservingAttack.make_pred(final_adversarial_sample, classifier) == 1: 
                    final_adversarial_sample[j] +=1
                    break
                   
        logger.info(f"Final Adversarial sample: {final_adversarial_sample} ")
        return final_adversarial_sample


    def generate_adversarial_sample(self, features, idx, candidates, perturbation, classifier):
        samples = candidates
        adversarial_samples = []
        l2_distances = []
        successful_idx = []
        feature_diffs = [] 
        for i, sample in enumerate(samples):  
            adv, adversarial_indices = FunctionalityPreservingAttack().make_initial_adversarial(sample, perturbation, classifier)
            if adv is not None:
                adversarial_sample, final_adv_indices = FunctionalityPreservingAttack().decrement_list(adv, perturbation, adversarial_indices, classifier)
                final_adversarial_sample = FunctionalityPreservingAttack().optimize(sample, adversarial_sample,final_adv_indices, classifier)
                FunctionalityPreservingAttack.feature_changes(sample, final_adversarial_sample, features)
                l2 = np.linalg.norm(final_adversarial_sample-sample)
                logger.info(f"l2 distance: {l2}")
                l2_distances.append(float(l2))
                successful_idx.append(idx[i]) 
                adversarial_samples.append(final_adversarial_sample)      
                
                np_sample = np.array(sample) 
                np_adv_sample = np.array(final_adversarial_sample) 
                diff_sample_l0 = np.count_nonzero(np_adv_sample != np_sample)
                feature_diffs.append(diff_sample_l0)          
                

                logger.info(f"sample number: {i}/{len(samples)} complete")
            
        logger.info(f"L2 distances: {l2_distances}")    
        
        average_l2 = sum(l2_distances)/len(l2_distances) 
        
        print(f"The average  l2 distance {average_l2} across {len(l2_distances)} samples")
        print(f"The biggest  l2 distance {max(l2_distances)} across {len(l2_distances)} samples")
        print(f"The smallest l2 distance {min(l2_distances)} across {len(l2_distances)} samples")
        
        average_feature_diff = sum(feature_diffs)/len(feature_diffs) 
                
        print(f"The average amount of features changed: {average_feature_diff}")
        print(f"The biggest amount of features changed: {np.max(feature_diffs)}")
        print(f"The least amount of features changed: {np.min(feature_diffs)}")
        
        return adversarial_samples, successful_idx
