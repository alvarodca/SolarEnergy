# Extracting the amount of defects in silicon wafers through SRH lifetime curves with machine learning models.

Bulk defects are highly critical on the efficiency of silicon solar cells. Bulk defects follow the Shockley-Read-Hall recombination statistics. Currently, there can be one single defect, two single defect, or one component with two level defects, however, they are hard to distinguish. This problem ought to be tackled due to the importance they have and the potential problems this could be solved. Because of this, a machine learning model is created to solve this.

I am currently a student experimenting with a simple understanding and knowledge on the topic of Neural Networks, I know a bit about defects in silicon wafers due to my father's studies. This repository is just for experimenting and learning a bit more about these topics, I am no expert in any of these areas.

**Generating the dataset**

This is all covered in the file dataset.py

The formula for calculating SRH lifetime curves depends on the present bulk defects. 

These is the formula for one defect curves
![alt text](image.png)

These SRH curves are calculated through different parameters which have been studied and we can provide valid intervals for them. As there can be some variations, Gaussian noises were added to incorporate a bit of randomness.

The parameters used were:
- E_t <- Defect Energy Level [-0.55,0.55]
- sigma_n, sigma_p <-Capture cross-sections [10^-17, 10^-13] 
- delta_n <-  Excess carrier concentration [10^13, 10^17]
- T <- Temperature. Values may vary but usually between 200 and 400
- N_t <- Defect Density [10^12]
- N_dop <- Doping Density [5.1 * 10^15]

Moreover, other parameters such as electron mass in Si or thermal velocities are covered.

Other parameters in the formula:

![alt text](image-1.png)

For two single defect, the SRH formula is given by 1/total_srh = 1/first_component_srh + 1/second_component_srh

For multilevel defects:
![alt text](image-2.png)

These parameters and calcualtions were randomized, simulated and created, each curve is comprised of 100 points. The dataset stored the value of carrier excess concentration, the srh obtained at that point, the temperature, the curve to which the point belonged and finally the label {0,1,2} depending on the defect in order to be able to train the model.

In the end, this enabled us to generate a dataset where each 100 points form a different SRH lifetime curve. Moreover, this generates an equally class balanced dataset with the same number of instances for all of them.

**Logistic Regression Model**

Covered in file logisticregression.py

Due to some problems when training our CNN model, a LR model was tested to verify whether we had a simple or too complex model or if our generated data was not good. 

As this model is too simple, we were not expecting a good result, but we wanted a result over 33% accuracy as this would show the model is better than random guessing and therefore is actually learning something.

This model was trained on a dataset made of 180000 curves, with approximately 60000 curves of each class. They were all created on the same temperature (300). The returned accuracy was 0.47, which suggested the model was indeed learning something and our data was actually correct. The main problem was our CNN model.


