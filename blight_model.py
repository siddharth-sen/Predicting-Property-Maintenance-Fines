# python3

import pandas as pd
import numpy as np

def blight_model():
    import pandas as pd
    import numpy as np

    df_train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
    df_train = df_train.where((df_train['compliance'] == 0) | (df_train['compliance'] == 1)).dropna(how='all')
    df_test = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    
    X = df_train.loc[:, ['ticket_id', 'agency_name', 'violation_street_number', 
       'violation_street_name', 'violation_zip_code', 'mailing_address_str_number',
       'mailing_address_str_name', 'city', 'state', 'zip_code',
       'non_us_str_code', 'country', 'violation_code', 'violation_description', 
       'disposition', 'fine_amount', 'admin_fee', 
       'state_fee', 'late_fee', 'discount_amount',
       'clean_up_cost', 'judgment_amount']].set_index('ticket_id')

    categorical_data = ['agency_name', 'state', 'violation_code',
                        'violation_description', 'disposition']
    numeric_data = ['fine_amount', 'admin_fee', 
                    'state_fee', 'late_fee', 'discount_amount',
                    'clean_up_cost', 'judgment_amount']

    X_cat = X.loc[:, categorical_data]
    X_cat = pd.get_dummies(X_cat)
    X_num = X.loc[:, numeric_data]

    X = pd.merge(X_cat, X_num, how='inner', left_index=True, right_index=True)
    y = df_train.loc[:, 'compliance']
    
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

    
    
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 8, random_state = 0).fit(X_train, y_train)

    feat_imp = pd.DataFrame({'Feature': list(X.columns),
                             'Importance': list(clf.feature_importances_)}).sort_values(by='Importance', ascending=False)

    
    N=5
    topN_features = list(pd.Series(feat_imp['Feature'][0:N]))
    
    
    from sklearn.ensemble import RandomForestClassifier

    X_top = X.loc[:, topN_features]
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, train_size=0.9, random_state=0)

    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0).fit(X_train, y_train)
    
    
    from sklearn.metrics import confusion_matrix

    y_pred = clf_rf.predict(X_test)
    y_prob_rf = clf_rf.predict_proba(X_test)
    y_prob = np.hsplit(y_prob_rf, 2)[1]

    confusion = confusion_matrix(y_test, y_pred)
    
    
    X_ = df_test.loc[:, ['ticket_id', 'agency_name', 'violation_street_number', 
       'violation_street_name', 'violation_zip_code', 'mailing_address_str_number',
       'mailing_address_str_name', 'city', 'state', 'zip_code',
       'non_us_str_code', 'country', 'violation_code', 'violation_description', 
       'disposition', 'fine_amount', 'admin_fee', 
       'state_fee', 'late_fee', 'discount_amount',
       'clean_up_cost', 'judgment_amount']].set_index('ticket_id')

    categorical_data = ['agency_name', 'state', 'violation_code',
                        'violation_description', 'disposition']
    numeric_data = ['fine_amount', 'admin_fee', 
                    'state_fee', 'late_fee', 'discount_amount',
                    'clean_up_cost', 'judgment_amount']

    X_cat_ = X_.loc[:, categorical_data]
    X_cat_ = pd.get_dummies(X_cat_)
    X_num_ = X_.loc[:, numeric_data]

    
    
    X_ = pd.merge(X_cat_, X_num_, how='inner', left_index=True, right_index=True)

    X_top_test = X_.loc[:, topN_features]
    
    y_pred_ = clf_rf.predict(X_top_test)
    y_prob_rf_ = clf_rf.predict_proba(X_top_test)
    y_prob_ = np.hsplit(y_prob_rf_, 2)[1].reshape(-1, )

    y_pred_prob = pd.Series(data=y_prob_, index=X_top_test.index, name='compliance')
    
    return y_pred_prob


