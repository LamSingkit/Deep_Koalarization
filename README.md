# E4040.2021Spring.OJBK.Project.SL4830.YL4578
## Discription
This is the final project of group OJBK, by Shengjie Lin and Yaoxuan Liu.  
  
E4040.2021Spring.OJBK.report.SL4830.YL4578.pdf is the final report.  
Final Project OJBK.ipynb is the only jupyter notebook.  

To **Train** the model, download the dataset(500MB) from [HERE](http://data.csail.mit.edu/places/places365/val_256.tar), place it in the Dataset folder. Run the codes in Training section of Jupyter notebook.  
To **Run** the trained model and predict your picture, download the model_weights(700MB) from [HERE](https://drive.google.com/file/d/1X3rKKbXVv5en_ztab_vdoRx4W1g57smq/view?usp=sharing) and run the codes in Prediction section of Jupyter notebook.  
  
## Files Discription
<pre>
Final Project OJBK.ipynb: 	Main Jupyter Notebook  
utils.py: 			Class and function used in Jupyter Notebook  
Figures: 			Proof of project run on GCP  
Prediction: 			Image used in prediction section of the notebook  
Results: 			Training history and prediction results  
</pre>
## File tree  
Folder PATH listing  
<pre>
|   Final Project OJBK.ipynb  
|   README.md  
|   utils.py  
|   E4040.2021Spring.OJBK.report.SL4830.YL4578.pdf  
|  
+---.ipynb_checkpoints  
|       Final Project OJBK-checkpoint.ipynb  
|         
+---Dataset  
|   \---val_256  
+---Figures  
|       Proof_1.png  
|       Proof_2.png  
|       Proof_3.png  
|         
+---model_weights  
|	saved_model.pd  
|	\---variables  
|		variables.data-00000-of-00002  
|		variables.data-00001-of-00002  
|		variables.index  
|  
+---Prediction  
|       im1.png  
|       im1021.jpg  
|       im1079.jpg  
|       im2.png  
|       im22604.jpg  
|       im233.jpg  
|       im3.png  
|       im591.jpg  
|         
\---Results  
        1.png  
        2.png  
        3.png  
        4.png  
        First 10 epoch.png  
        Next 8 epoch.png  
        
</pre>
