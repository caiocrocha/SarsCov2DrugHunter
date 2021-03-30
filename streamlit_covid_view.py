# Usar postera inteiro como conjunto de treinamento (opcao, elimina teste)
# Upload dados de treinamento, teste (generalizacao)

import os
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot

from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# Program
class App():
    def __init__(self, DATA_URL):
        # Show logo, title and description
        self.show_logo()
        self.show_description()
        # Check .metadata directory
        self.metadata_dir()

        ########################
        # Load Activity data 
        ########################
        st.markdown('## **Activity data**')
        st.markdown('### Visualizing properties')
        self.downloaded_data = self.download_activity(DATA_URL)
        self.write_smiles(self.downloaded_data, '.metadata/smiles.smi')

        #######################
        # Summary of the data 
        #######################
        self.data = self.downloaded_data.copy()
        self.activity_label = None
        self.show_properties() # show properties and set activity label
        self.label_compounds() # drop activators and label the compounds according to their activity

        #######################
        # Load descriptors 
        #######################
        self.calc = None
        self.descriptors_cols = None
        self.descriptors      = self.calculate_descriptors()
        self.merged_data      = None
        
        #######################
        # ML 
        #######################
        self.pca = None
        self.new_data = None
        self.pipeline = None
        self.X_train  = None
        self.X_test   = None
        self.y_train  = None
        self.y_test   = None
        self.test_proba  = None
        self.train_proba = None
    
    # Functions
    @staticmethod
    def show_logo():
        st.sidebar.image('logo/Logo_medium.png')

    @staticmethod
    def show_description():
        st.markdown('''## **Welcome to**
# SARS-CoV-2
## Machine Learning Drug Hunter
A straightforward App that combines experimental activity data, molecular descriptors and machine 
learning for classifying potential drug candidates against the SARS-CoV-2 Main Protease (MPro).     

We use the **COVID Moonshot**, a public collaborative initiatiave by **PostEra**, as the dataset of 
compounds containing the experimental activity data for the machine learning classifiers. We'd like 
to express our sincere thanks to PostEra, without which this work wouldn't have been possible.    

The molecular descriptors can be automatically calculated with Mordred or RDKit, or you can also 
provide a CSV file of molecular descriptors calculated with an external program of your preference.     

This main window is going to guide you through the App, while the sidebar to the left offers you an extra 
interactive experience with options that allow more control over the construction of the Pipeline. **Let's get started!**
''')

    @staticmethod
    def metadata_dir():
        if st.sidebar.checkbox('Clear ".metadata" directory'):
            st.sidebar.warning('''You are about to erase all the contents of **.metadata**. This will remove 
            any saved files, so this action is only recomended if you encountered an error. Are you 
            sure you want to proceed?''')
            st.sidebar.write('_If do not want to delete the contents of **.metadata**, just unselect the checkbox above._')
            if st.sidebar.button('Yes, I want to delete the contents of the ".metadata" folder'):
                if os.path.isdir('.metadata'):
                    try:
                        import shutil
                        shutil.rmtree('.metadata')
                        st.caching.clear_cache()
                        st.sidebar.success('Directory successfully cleared!')
                    except OSError as e:
                        st.error('Could not remove folder. Error traceback: ')
                        st.error(str(e))
            
        if not os.path.isdir('.metadata'):
            try:
                os.mkdir('.metadata')
            except OSError as e:
                st.error('Could not create **.metadata** directory. Error traceback: ')
                st.error(str(e))
                st.stop()
    
    @staticmethod
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def download_activity(DATA_URL):
        # Verbose
        st.text('Fetching the data from PostEra...')
        st.markdown('Loading activity data...')
        data = pd.read_csv(DATA_URL)
        if not os.path.isdir('.metadata/csv'):
            try:
                os.mkdir('.metadata/csv')
            except OSError as e:
                st.error('Could not create ".metadata/csv". Error traceback: ')
                st.error(str(e))

        data.to_csv('.metadata/csv/activity.csv', index=False)
        st.text('Data saved to ".metadata/csv/activity.csv"')
        return data
    
    @staticmethod
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def write_smiles(data, smiles):
        # Write smiles to disk
        data[['SMILES','CID']].to_csv(smiles, sep='\t', header=None, index=False)

    @staticmethod
    def write_mordred_descriptors(smiles, csv):
        # Run MORDRED with smiles file.
        # Could not make mordred work with rdkit.
        # There is a lot of (unsolved) confusion with library compatibility
        # Let's rely on the manual way
        if os.path.isfile(smiles) and not os.path.isfile(f'{csv}.gz'):
            os.system(f'python -m mordred {smiles} > {csv}')

        if os.path.isfile(csv):
            os.system(f'gzip {csv}')
    
    @staticmethod
    def write_rdkit_descriptors(smiles, csv, data):
        if os.path.isfile(smiles) and not os.path.isfile(f'{csv}.gz'):
            # Get molecules from SMILES
            mols = [Chem.MolFromSmiles(i) for i in data['SMILES']]

            # Get list of descriptors
            descriptors_list = [a[0] for a in Chem.Descriptors.descList]

            calculator = MolecularDescriptorCalculator(descriptors_list)
            calc_descriptors = [calculator.CalcDescriptors(m) for m in mols]
            
            descriptors = pd.DataFrame(calc_descriptors, columns=descriptors_list)
            descriptors.insert(0, column='CID', value=data['CID'].tolist())
            descriptors.to_csv(f'{csv}.gz', index=False, compression='gzip')

    def calculate_descriptors(self):
        st.markdown("## **Descriptors**")
        if st.checkbox('Calculate Mordred descriptors'):
            self.write_mordred_descriptors('.metadata/smiles.smi', '.metadata/csv/mordred.csv')
            # Read MORDRED descriptors
            descriptors = pd.read_csv('.metadata/csv/mordred.csv.gz', compression='gzip')
            descriptors.rename(columns={'name':'CID'}, inplace=True)
            self.calc = 'mordred' # control variable
        elif st.checkbox('Calculate RDKit descriptors'):
            self.write_rdkit_descriptors('.metadata/smiles.smi', '.metadata/csv/rdkit.csv', self.data)
            # Read RDKit descriptors
            descriptors = pd.read_csv('.metadata/csv/rdkit.csv.gz', compression='gzip')
            self.calc = 'rdkit' # control variable
        else:
            file = st.file_uploader('or Upload descriptors file')
            show_file = st.empty()

            if not file:
                show_file.info("Please upload a file of type: .csv")
                st.stop()
            else:
                descriptors = pd.read_csv(file)
                if not 'CID' in descriptors.columns:
                    st.error('Compounds must be identified by "CID"')
                    st.stop()
            file.close()
            self.calc = 'other' # control variable
        
        st.dataframe(descriptors.head())

        self.descriptors_cols = descriptors.columns.tolist()[1:]
        selected = st.multiselect(label="Select descriptors", options=(
            ['Select all ({})'.format(len(self.descriptors_cols))] + self.descriptors_cols))
        if 'Select all ({})'.format(len(self.descriptors_cols)) in selected:
            selected = self.descriptors_cols
        st.write("You have selected", len(selected), "features")

        if not selected:
            st.stop()
        
        descriptors = descriptors[['CID'] + selected]
        return descriptors
        
    def show_properties(self):
        # List numeric columns
        data_numeric = self.data.select_dtypes(include=[int,float]).columns.tolist()
        if 'activity' in data_numeric:
            data_numeric.remove('activity')

        ########################
        # Explore data
        ########################

        st.sidebar.header('Activity data')
        # Create a sidebar dropdown to select property to show.
        activity_label = st.sidebar.selectbox(label="Filter by: *",
                                        options=([None, *data_numeric]))
        st.sidebar.markdown('''\* _The classifier will be trained according to the selected property. 
If no property is selected, then **f_inhibition_at_50_uM** will be used for labeling the compounds.    
A compound will be considered active if the **`Selected Property > 50`**. This value can be adjusted with the slider below._''')
        
        if activity_label is None:
            activity_label = 'f_inhibition_at_50_uM'

        self.activity_label = activity_label

        # Create a sidebar slider to filter property
        ## Step 1 - Pick min & max for picked property 
        max_val  = float(self.data[activity_label].max())
        min_val  = float(self.data[activity_label].min())
        #mean_val = float(self.data[activity_label].mean())

        ## Step 2 - Create the sidebar slider
        min_filter, max_filter = st.slider("Filter by: " + activity_label, 
                                min_val,
                                max_val,
                                (min_val, max_val))
        
        df_properties = self.data[['CID', activity_label]].dropna()
        df_filtered = df_properties[df_properties[activity_label].between(
            float(min_filter), float(max_filter))]
        mean_filter = float(df_filtered[activity_label].mean())

        st.markdown(f'''
        | Property | Min | Max | Mean |
        | --- | --- | --- | --- |
        | {activity_label} | {min_filter:2g} | {max_filter:2g} | {mean_filter:2g} |

        ''')

        st.text('')
        if st.checkbox('Show downloaded data'):
            st.dataframe(self.downloaded_data)

    def label_compounds(self):
        threshold = st.sidebar.slider("Threshold for selecting active compounds:", 0, 100, value=50)

        # Plot the distribution of the data
        dist = self.downloaded_data[['CID', self.activity_label]].copy()
        dist['activity'] = 'inhibitor'
        dist.loc[dist[self.activity_label] <= threshold, 'activity'] = 'inactive'
        dist.loc[dist[self.activity_label] < 0, 'activity'] = 'activator'

        if not st.checkbox('Hide graph'):
            fig, ax = pyplot.subplots(figsize=(15,5))
            sns.histplot(data=dist, x='f_inhibition_at_50_uM', hue='activity', ax=ax)
            pyplot.ylabel('Number of compounds')
            pyplot.title('Distribution of the data')
            st.pyplot(fig)
        
        self.data.dropna(subset=[self.activity_label], inplace=True)
        self.data = self.data.query(f'{self.activity_label} > 0') # Drop activators (negative inhibition)

        # Label the compounds
        self.data['activity'] = 0
        self.data.loc[self.data[self.activity_label] > threshold, 'activity'] = 1

        st.write('Note: All **activators** have been removed from the dataset, and the **inhibitors** will be referred as **active** compounds.')
        # Create sublists
        actives    = self.data.query(f'{self.activity_label} > {threshold}')
        inactives  = self.data.query(f'{self.activity_label} <= {threshold}')

        st.markdown(f'''
        |Compounds|Active|Inactive|
        |---|---|---|
        |{len(self.data)}|{len(actives)}|{len(inactives)}|
        ''')
        
    def merge_dataset(self):
        # Merge the dataset to include activity data and descriptors.
        merged_data = pd.merge(self.data[['CID', self.activity_label, 'activity']].dropna(), 
                            self.descriptors, on=['CID'])
        # Write Merged Dataset
        if not os.path.isfile('.metadata/csv/merged.csv'):
            merged_data.to_csv('.metadata/csv/merged.csv', index=False)

        return merged_data

    def calculate_cross_corr(self):
        X = self.merged_data.drop(['CID','activity'], axis=1).dropna(axis=1)
        Y = self.merged_data[self.activity_label]
        corr = X.corr()
        st.write(corr.head(5))

        if st.checkbox('Show entire DataFrame'):
            if len(corr) <= 100:
                st.write(corr)
            else:
                st.error("Sorry, large DataFrames can't be displayed!")

        if st.checkbox('Show correlation HeatMap'):
            if len(corr) <= 100:
                fig, ax = pyplot.subplots(figsize=(10,10))
                sns.heatmap(corr, annot=True, cmap='Reds', square=True, ax=ax)
                st.pyplot(fig)
            else:
                st.error("Sorry, large DataFrames can't be displayed!")

        if st.checkbox('Remove highly correlated features (|Correlation| > Correlation Threshold)', True):
            value = st.slider('Correlation Threshold', 0.0, 1.0, value=0.95)

            # https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
            # Create correlation matrix
            corr_matrix = corr.drop([self.activity_label], axis=1).abs()
            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

            # Find features with correlation greater than "value"
            to_drop = [column for column in upper.columns if any(upper[column] > value)]

            # Drop features 
            st.write('Removed features: ')
            st.write(to_drop)
            self.descriptors.drop(to_drop, axis=1, inplace=True)
            self.merged_data.drop(to_drop, axis=1, inplace=True)

    def calculate_pca(self):
        max_value = len(self.descriptors_cols)
        default = 0.9
        n_components = st.number_input(f'Please enter the number of components to select [0, {max_value}]: ', 
                        value=default, min_value=0.0, max_value=float(max_value))
        st.markdown(f'''\* If the input number is less than 1, then it will correspond to the percentage of the explained 
variance. E.g. the default value corresponds to an explained variance of {default * 100}%.''')
        if n_components > 1:
            n_components = int(n_components)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from imblearn.pipeline import make_pipeline

        # Split training set into X and y
        y = self.merged_data['activity']
        X = self.merged_data[self.descriptors_cols].copy()

        pca = make_pipeline(StandardScaler(), PCA(n_components=n_components))

        state = st.text('Running PCA...')
        # Fit and transform the training data
        X_pca = pca.fit_transform(X)
        self.pca = pca

        state.text('PCA completed!')
        variance_total = sum(pca['pca'].explained_variance_ratio_)
        if pca['pca'].n_components_ < 51:
            fig, ax = pyplot.subplots(figsize=(12,4))
            sns.barplot(x=[i for i in range(1, pca['pca'].n_components_ + 1)], y=pca['pca'].explained_variance_ratio_, ax=ax)
            ax.set(xlabel='Principal Component', ylabel='Explained variance ratio', 
                                    title=f'Variance explained by {variance_total * 100:.1f}%')
            st.pyplot(fig)
        else:
            st.write(f'Explained variance: {variance_total * 100:.1f}%')

        # Reassign the data to the new transformed data
        pca_data = pd.DataFrame(X_pca)
        pca_features = [f'PCA_{i:02d}' for i in range(1, pca['pca'].n_components_ + 1)]
        pca_data.columns = pca_features
        pca_data['CID'] = self.merged_data['CID'].tolist()
        pca_data['activity'] = y.tolist()
        # Rearrange the columns
        cols = pca_data.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        pca_data = pca_data[cols]

        self.merged_data = pca_data
        self.descriptors = pca_data[['CID'] + pca_features]

        st.write('### Principal Components')
        st.write(self.descriptors.head())

    def feature_selection(self):
        st.markdown('# Feature selection')
        st.markdown('Filter the selected descriptors. The steps bellow are applied sequentially.')

        st.markdown('## Cross Correlation')
        if st.checkbox('Compute the cross correlation between the features'):
            self.calculate_cross_corr()

        st.markdown('## PCA')
        if st.checkbox('Calculate PCA of the selected features'):
            self.calculate_pca()
        
        st.write('## Model input features')
        st.write(self.descriptors.columns.tolist()[1:])

        if st.checkbox('Show histogram plots of the selected features'):
            descriptors_list = self.descriptors.columns.tolist()[1:]
            tmp = pd.melt(self.descriptors, id_vars=['CID'], value_vars=descriptors_list[:12])
            g = sns.FacetGrid(data=tmp, col='variable', col_wrap=4, sharey=False, sharex=False)
            g.map(sns.histplot, 'value')
            if len(descriptors_list) > 11:
                st.warning("""Unfortunately, we can't plot all selected descriptors    
Showing the distribution plots of the top 12 features""")
            st.pyplot(g)
            
    @staticmethod
    def select_model():
        model_list = ['RandomForestClassifier', 'XGBClassifier', 'LogisticRegression', 'LinearSVC']
        model_name = st.selectbox(label="Classifier", options=model_list)
        st.markdown('''_The default hyperparameters are the optimal parameters found in our study, but feel free 
to change them whenever you want in the sidebar beside.
The constructed model is a **Pipeline** of _**`ColumnTransformer + SMOTE`**_, which automatically transforms the input data to the classifier._''')

        st.sidebar.header('Classifier')
        st.sidebar.subheader(model_name)
        #st.sidebar.markdown('''Note: The hyperparaters showed bellow are the optimal parameters found in our study. 
#Nevertheless, feel free to change them as you will.''')
        if model_name == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier

            n_estimators = st.sidebar.slider("Number of Estimators", 0, 1000, value=500)
            max_depth = st.sidebar.slider("Maximum depth per Tree", 0, 10, value=6)
            return (model_name, RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=13))

        elif model_name == 'XGBClassifier':
            from xgboost import XGBClassifier

            n_estimators = st.sidebar.slider("Number of Estimators", 0, 1000, value=200)
            max_depth = st.sidebar.slider("Maximum Depth per Tree", 0, 10, value=3)
            eta = st.sidebar.slider("Learning Rate (ETA)", 0.0, 1.0, value=0.1)
            return (model_name, XGBClassifier(objective='reg:logistic', n_estimators=n_estimators, 
                max_depth=max_depth, eta=eta, random_state=13))

        elif model_name == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            solver = st.sidebar.selectbox(label="Solver", options=['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'])
            max_iter = st.sidebar.slider("Maximum number of iterations", 100, 6000, step=100, value=100)
            return (model_name, LogisticRegression(solver=solver, 
                                                    max_iter=max_iter, random_state=13))

        elif model_name == 'LinearSVC':
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            # CalibratedClassifierCV applies probability transformation 
            # on top of the SVC outputs, so we can plot its ROC curve
            # https://stackoverflow.com/a/39712590/13131079
            max_iter = st.sidebar.slider("Maximum number of iterations", 100, 6000, step=100, value=100)
            return (model_name, CalibratedClassifierCV(
                base_estimator=LinearSVC(dual=False, max_iter=max_iter, random_state=13)))

    def split_X_and_y(self):
        from sklearn.model_selection import train_test_split
        X = self.merged_data[self.descriptors.columns[1:]]
        y = self.merged_data['activity']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                            X, y, test_size=0.2, random_state=27)

    @st.cache(suppress_st_warning=True)
    def mlpipeline(self, model_name, model):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import OneHotEncoder
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline
        
        transformer = ColumnTransformer(transformers=[
            ('continuous', StandardScaler(), self.X_train.select_dtypes(include=float).columns.tolist()), 
            ('discrete', OneHotEncoder(handle_unknown='ignore'), self.X_train.select_dtypes(include=int).columns.tolist())
        ])

        self.pipeline = Pipeline(steps=[('smote', SMOTE(random_state=42)), 
                                        ('transformer', transformer), 
                                        ('clf', model)
                                        ])
        self.pipeline.fit(self.X_train, self.y_train)

        import pickle
        # Serialize model
        if not os.path.isdir('pickle'):
            os.mkdir('pickle')
        with open(f'pickle/{model_name}.pickle', 'wb') as file:
            pickle.dump(self.pipeline, file)

        features = list(self.descriptors.columns[1:])
        # Save input features names
        with open('.metadata/features.lst', 'w+') as features_file:
            features_file.write("\n".join(features))
    
    def train_test_scores(self, model_name):
        import pickle
        try:
            file = open(f'pickle/{model_name}.pickle', 'rb')
            self.pipeline = pickle.load(file)
            file.close()
        except OSError as e:
            st.error(f"Oops! It seems the model hasn't been trained yet. Error traceback: ")
            st.error(str(e))
            st.stop()

        from sklearn.metrics import roc_curve, auc
        fig, ax = pyplot.subplots()

        try:
            self.test_proba = self.pipeline.predict_proba(self.X_test)[:,1]
            self.train_proba = self.pipeline.predict_proba(self.X_train)[:,1]
        except ValueError as e:
            st.error('Expected features do not match the given features. Please train the model again. Error traceback: ')
            st.error(str(e))
            st.stop()

        fpr, tpr, _ = roc_curve(self.y_test, self.test_proba)
        auc_test = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Test set: {auc_test:>.2f}')

        fpr, tpr, _ = roc_curve(self.y_train, self.train_proba)
        auc_train = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Training set: {auc_train:>.2f}')

        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title(model_name)
        pyplot.legend()

        if not os.path.isdir('.metadata/roc'):
            os.makedirs('.metadata/roc')
        pyplot.savefig(f'.metadata/roc/{model_name}.png', dpi=200)
        st.markdown('### Receiver Operating Characteristic')
        st.pyplot(fig)

        if st.checkbox('Show ROC of the previous models'):
            _, _, filenames = next(os.walk('.metadata/roc'))
            filenames.remove(f'{model_name}.png')
            for clf in filenames:
                st.image(f'.metadata/roc/{clf}')
            
        from sklearn.metrics import f1_score
        from imblearn.metrics import geometric_mean_score
        
        y_pred = self.pipeline.predict(self.X_test)
        y_pred_train = self.pipeline.predict(self.X_train)
        scores = [model_name, f1_score(self.y_test, y_pred), geometric_mean_score(self.y_test, y_pred), auc_test, 
            f1_score(self.y_train, y_pred_train), geometric_mean_score(self.y_train, y_pred_train), auc_train]
        
        scores_data = pd.DataFrame([scores], columns=['Classifier','test_f1','test_geometric_mean','test_roc_auc', 
                'train_f1','train_geometric_mean','train_roc_auc'])
        if os.path.isfile('.metadata/scores.csv'):
            scores_data = pd.concat([scores_data, pd.read_csv('.metadata/scores.csv')])
            scores_data.drop_duplicates(subset=['Classifier'], inplace=True, keep='last')
            
        scores_data.to_csv('.metadata/scores.csv', index=False)
        st.write('### Scoring metrics')
        st.write(scores_data)
    
    def upload_new_compounds(self):
        st.markdown('## Classify new compounds')
        file = st.file_uploader('Upload file *')
        show_file = st.empty()
        st.markdown('''\* File must contain the following columns:   
1 - "SMILES": SMILES structures of the compounds     
2 - "CID": compounds ID''')

        if not file:
            show_file.info("Please upload a file of type: .csv")
            st.stop()
        else:
            self.new_data = pd.read_csv(file)
            st.write(self.new_data.head())
        file.close()
        
        self.write_smiles(self.new_data, '.metadata/smiles2.smi')
        if self.calc == 'mordred':
            self.write_mordred_descriptors('.metadata/smiles2.smi', '.metadata/csv/mordred2.csv')
            # Read MORDRED descriptors
            descriptors = pd.read_csv('.metadata/csv/mordred2.csv.gz', compression='gzip')
            descriptors.rename(columns={'name':'CID'}, inplace=True)
        elif self.calc == 'rdkit':
            self.write_rdkit_descriptors('.metadata/smiles2.smi', '.metadata/csv/rdkit2.csv', self.new_data)
            # Read RDKit descriptors
            descriptors = pd.read_csv('.metadata/csv/rdkit2.csv.gz', compression='gzip')
        else:
            file = st.file_uploader('Upload the descriptors file for the new compounds')
            show_file = st.empty()

            if not file:
                show_file.info("Please upload a file of type: .csv")
                st.stop()
            else:
                descriptors = pd.read_csv(file)
                if not 'CID' in descriptors.columns:
                    st.error('Compounds must be identified by "CID"')
                    st.stop()
            file.close()
            try:
                tmp = pd.merge(self.new_data, descriptors[['CID'] + self.descriptors_cols], on=['CID'])
            except KeyError as e:
                st.error('''Expected features do not match the given features. 
Please make sure that the input file contains the same descriptors used for training the model.''')
                st.stop()
        
        descriptors.dropna(subset=self.descriptors_cols, inplace=True)
        
        if self.pca is not None:
            X = descriptors[self.descriptors_cols]
            X_new = self.pca.transform(X)
            # Reassign the data to the new transformed data
            pca_data = pd.DataFrame(X_new)
            pca_features = [f'PCA_{i:02d}' for i in range(1, self.pca['pca'].n_components_ + 1)]
            pca_data.columns = pca_features
            pca_data['CID'] = descriptors['CID'].tolist()
            # Rearrange the columns
            cols = pca_data.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            pca_data = pca_data[cols]
            self.new_data = pca_data
        else:
            self.new_data = pd.merge(self.new_data, descriptors[['CID'] + self.descriptors_cols], on=['CID'])
            
    def pipeline_predict(self):
        st.markdown('### **Predictions**')
        features = self.descriptors.columns[1:].tolist()
        X_val = self.new_data[features]

        y_pred  = pd.Series(self.pipeline.predict(X_val))
        y_proba = self.pipeline.predict_proba(X_val)[:,1]

        predictions = self.new_data[['CID']].copy()
        predictions['prediction'] = y_pred.replace({1: 'active', 0: 'inactive'}).tolist()
        predictions['probability'] = y_proba
        predictions.sort_values('probability', ascending=False, inplace=True)

        counts = predictions['prediction'].value_counts()

        st.markdown(f'''
        |Compounds|Active|Inactive|
        |---|---|---|
        |{len(predictions)}|{counts['active']}|{counts['inactive']}|
        ''')

        st.write('')
        st.write('Top compounds:')
        st.write(predictions.reset_index(drop=True).head())

        predictions.to_csv('predictions.csv', index=False)
        st.write('Predictions saved to "predictions.csv".')

    @staticmethod
    def copyright_note():
        st.markdown('----------------------------------------------------')
        st.markdown('Copyright (c) 2021 CAIO C. ROCHA, DIEGO E. B. GOMES')
        st.markdown('Definir/atualizar copyright quando estiver pronto')


def main():
    # """Run this function to display the Streamlit app"""
    # st.info(__doc__)

    # Config
    DATA_URL = ('https://covid.postera.ai/covid/activity_data.csv')
    app = App(DATA_URL)

    if app.descriptors is not None:
        app.merged_data = app.merge_dataset()
        app.feature_selection()

        st.write('# Machine learning')
        app.split_X_and_y()
        model_name, model = app.select_model()
        if st.checkbox('Train model'):
            app.mlpipeline(model_name, model)
            st.markdown(f'_Model saved to pickle/{model_name}.pickle_')
        
        app.train_test_scores(model_name)
        app.upload_new_compounds()
        app.pipeline_predict()
        
    # Copyright footnote
    app.copyright_note()

if __name__== '__main__': main()
