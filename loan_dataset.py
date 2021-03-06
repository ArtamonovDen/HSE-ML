import pandas as pd
import torch
from torch.functional import norm

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



class VehicleLoanDataset(Dataset):
    def __init__(self, csvpath, balance=True, normalize=False, mode='train', test_size = 0.2, random_state=42):
        raw_data = pd.read_csv(csvpath)  

        self.mode = mode
        self.normalize = normalize
        data = self._preprocess_data(raw_data)
        data = self._balance(data)
        self.data = data

        N,M = data.values.shape
        X =  data.values[:,:-1]
        y = data.values[:,-1].reshape(-1,1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size = test_size, random_state = random_state
            )

    def set_mode(self, mode):
        self.mode = mode

    def get_shape(self):
        if self.mode == 'train':
            return self.X_train.shape    
        else:
            return self.X_test.shape 

    def get_test_data(self):
        return self.X_test, self.y_test

    def __len__(self):
        if self.mode == 'train':
            return len(self.y_train)
        else:
            return len(self.y_test)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return torch.Tensor(self.X_train[idx]), torch.Tensor(self.y_train[idx])
        else:
            return torch.Tensor(self.X_test[idx]), torch.Tensor(self.y_test[idx])

    def _balance(self, data):
        d_one = data[data['loan_default'] == 1]
        N_one = d_one.shape[0]
        d_two = data[data['loan_default'] == 0].sample(N_one)
        data = d_one.append(d_two, ignore_index=True)
        return data

    def _preprocess_data(self, data):

        # Remove useless ID-columns
        data.drop(columns=['UniqueID', 'Current_pincode_ID', 'State_ID'], inplace=True)

        # Label Employment type
        data['Employment.Type'] = data['Employment.Type'].astype(str)
        employment_type_encoder = LabelEncoder()
        data['Employment.Type'] = employment_type_encoder.fit_transform(data['Employment.Type']) # TODO может какой-нибудь торчовый энкодер? 

        # Remove DisbursalDate as additional info
        data.drop(columns=['DisbursalDate'], inplace=True)

        # Transform Date.of.Birt to year of birth
        data['Date.of.Birth'] = pd.to_datetime(data['Date.of.Birth'])
        data['Date.of.Birth'] = data['Date.of.Birth'].apply(lambda x: x.replace(year=x.year-100) if x.year>2000 else x.replace(year=x.year))
        data['Date.of.Birth'] = data['Date.of.Birth'].apply(lambda x: x.year)


        # Remove PERFORM_CNS.SCORE.DESCRIPTION as PERFORM_CNS.SCORE contains same info
        data.drop(columns=['PERFORM_CNS.SCORE.DESCRIPTION'], inplace=True)

        # Transform to numeric features
        data['AVERAGE.ACCT.AGE'] = self._preprocess_to_month_num(data, 'AVERAGE.ACCT.AGE')
        data['CREDIT.HISTORY.LENGTH'] = self._preprocess_to_month_num(data, 'CREDIT.HISTORY.LENGTH')

        if self.normalize:
            numeric_features = ['disbursed_amount', 'asset_cost', 'ltv', 'Date.of.Birth','PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS',
                'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT',
                'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS',
                'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT',
                'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT', 'NEW.ACCTS.IN.LAST.SIX.MONTHS',
                'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'AVERAGE.ACCT.AGE',
                'CREDIT.HISTORY.LENGTH','NO.OF_INQUIRIES' ]
            data[numeric_features] = (data[numeric_features] - data[numeric_features].mean(axis = 0))/data[numeric_features].std(axis = 0)

        return data


    def _preprocess_to_month_num(self, data, col):
        df_ = data[col].str.extractall('(\d+)').unstack()
        df_.columns = df_.columns.droplevel(0)
        return df_.iloc[:,0].astype(int).mul(12) + df_.iloc[:,1].astype(int)
