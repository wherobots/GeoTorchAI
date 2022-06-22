
def are_dfs_equal(df1, df2):
    '''if df1.schema != df2.schema:
        return False'''
    if df1.collect() != df2.collect():
        return False
    return True
