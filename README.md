# Latent Mutual Feature Extraction for Cross-domain Recommendation (LFCDR)
This is the implementation of our paper **Latent Mutual Feature Extraction for Cross-domain Recommendation**, which has been accepted by Knowledge and Information Systems(2024). [Paper](https://doi.org/10.1007/s10115-024-02065-y)

The aim of this paper is to propose a Cross-domain Recommendation (CDR) model targeting heterogeneous domains. Previous studies have mainly focused on homogeneous domains and pose limitations when applied to heterogeneous domains without common users, items, and metadata. To overcome this challenge, we propose a heterogeneous CDR model called Latent Features Cross-domain Recommendation (LFCDR). Our model leverages Latent Features (LF), which construct the correlations between user and item features based on domain categories, where a category represents the domain attributes. By extracting the LF of each domain, we find similar domain latent features and improve the performance of the sparsity domain through transfer learning. We performed experiments on Latent Features Recommendation (LFR), a recommendation system using LF, and LFCDR, a CDR using LF of heterogeneous domains, using three heterogeneous domain datasets, and compared their performances with a Factorization Machine (FM). Our results illustrated that the performance of the LFR improved by up to 1.65, as measured by Mean Absolute Error (MAE), compared to the FM. Additionally, the performance of the LFCDR improved by up to 1.66, depending on the relevance of the domain's category.
