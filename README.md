# Tuberculosis_project
Tuberculosis (TB) is an infectious disease caused by the Mycobacterium tuber-
culosis (Mtb) bacillus [1]. Historically, this disease has been one of the deadliest
for humans. Every year, there are still more than 10 million new cases of active
tuberculosis worldwide and an estimated 1.3 million deaths [2]. Given this, it is
essential to find new drugs that act quickly, with a broader efficacy and minimal
side effects. Despite decades of research on immune responses that determine
protection against tuberculosis, there is still no clear idea of the set of immune
responses needed to prevent infection or the progression of the disease [3].
The rise of multidrug-resistant strains of Mtb has further compounded the
problem of TB control, highlighting the need for new and effective treatments.
For these reasons, computational methods, such as ML, can be potential alter-
native approaches to improve the accuracy and effectiveness of TB diagnosis and
treatment, while simultaneously reducing the costs, time and resources required
to manage the disease. 

This study attempts to identify novel proteins that could serve as potential
targets for combating Mtb. The methodology starts with data compilation from
different databases that contain information on drug-target interactions (DTI).
Then, leveraging state-of-the-art classifiers for DTI prediction, new interactions
will be classified and subsequently clustered with the entire dataset. These
clustered data will then undergo a selection of few points, which will be validated
against knowledge from literature.

### General information

In the "Code" folder we have 2 folders ("DeepDTA" e "DLM-DTI") that contain the code and documents necessary to obtain the results demonstrated in the article. 
Dataset "BindingDB_All_202406_tsv" exceeded the Github's file size limit of 100MB. Because of that, its possible extract the dataset from the following link: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_All_202406_tsv.zip

### References

[1] M. M. Ibrahim, T. M. Isyaka, U. M. Askira, J. B. Umar, M. A. Isa, et al.,
“Trends in the incidence of rifampicin resistant mycobacterium tuberculosis
infection in northeastern nigeria,” Scientific African, vol. 17, 9 2022.

[2] Global tuberculosis report 2023, 2023.

[3] J. A. L. Flynn and J. Chan, “Immune cell interactions in tuberculosis,” pp.
4682–4702, 12 2022.
