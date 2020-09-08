def catboost_target_encoder(train, test, categorical_columns, target):
    
    train_new = train.copy()
    test_new = test.copy()
    
    for column in categorical_columns:
        
        global_mean = train[target].mean()
        cumulative_sum = train.groupby(column)[target].cumsum() - train[target]
        cumulative_count = train.groupby(column).cumcount()
        
        train_new[column + "_encoding"] = cumulative_sum/cumulative_count
        train_new[column + '_encoding'].fillna(global_mean, inplace=True)


        mean_encoding = train_new.groupby(column)[column + '_encoding'].mean()
        test_new[column + "_encoding"]= test_new[column].apply(lambda x: mean_encoding[x] if x in mean_encoding else global_mean)

    return train_new, test_new
