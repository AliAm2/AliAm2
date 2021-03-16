.. code:: ipython3

    pip install mlxtend


.. parsed-literal::

    Collecting mlxtend
      Downloading mlxtend-0.18.0-py2.py3-none-any.whl (1.3 MB)
    [K     |████████████████████████████████| 1.3 MB 333 kB/s eta 0:00:01
    [?25hRequirement already satisfied: scikit-learn>=0.20.3 in /opt/anaconda3/lib/python3.8/site-packages (from mlxtend) (0.23.2)
    Requirement already satisfied: matplotlib>=3.0.0 in /opt/anaconda3/lib/python3.8/site-packages (from mlxtend) (3.3.2)
    Requirement already satisfied: scipy>=1.2.1 in /opt/anaconda3/lib/python3.8/site-packages (from mlxtend) (1.5.2)
    Requirement already satisfied: setuptools in /opt/anaconda3/lib/python3.8/site-packages (from mlxtend) (50.3.1.post20201107)
    Requirement already satisfied: numpy>=1.16.2 in /opt/anaconda3/lib/python3.8/site-packages (from mlxtend) (1.19.2)
    Requirement already satisfied: joblib>=0.13.2 in /opt/anaconda3/lib/python3.8/site-packages (from mlxtend) (0.17.0)
    Requirement already satisfied: pandas>=0.24.2 in /opt/anaconda3/lib/python3.8/site-packages (from mlxtend) (1.1.3)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/anaconda3/lib/python3.8/site-packages (from scikit-learn>=0.20.3->mlxtend) (2.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in /opt/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.0.0->mlxtend) (2.4.7)
    Requirement already satisfied: pillow>=6.2.0 in /opt/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.0.0->mlxtend) (8.0.1)
    Requirement already satisfied: certifi>=2020.06.20 in /opt/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.0.0->mlxtend) (2020.6.20)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.0.0->mlxtend) (1.3.0)
    Requirement already satisfied: cycler>=0.10 in /opt/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.0.0->mlxtend) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.0.0->mlxtend) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /opt/anaconda3/lib/python3.8/site-packages (from pandas>=0.24.2->mlxtend) (2020.1)
    Requirement already satisfied: six in /opt/anaconda3/lib/python3.8/site-packages (from cycler>=0.10->matplotlib>=3.0.0->mlxtend) (1.15.0)
    Installing collected packages: mlxtend
    Successfully installed mlxtend-0.18.0
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from mlxtend.frequent_patterns import apriori

.. code:: ipython3

    
    dataset = [['Drink', 'Nuts', 'Diaper'],
          ['Drink', 'Coffee', 'Diaper'],
          ['Drink', 'Diaper', 'Eggs'],
          ['Nuts', 'Eggs', 'Milk'],
          ['Nuts', 'Coffee', 'Diaper', 'Eggs', 'Milk']]

.. code:: ipython3

    dataset




.. parsed-literal::

    [['Drink', 'Nuts', 'Diaper'],
     ['Drink', 'Coffee', 'Diaper'],
     ['Drink', 'Diaper', 'Eggs'],
     ['Nuts', 'Eggs', 'Milk'],
     ['Nuts', 'Coffee', 'Diaper', 'Eggs', 'Milk']]



.. code:: ipython3

    
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    
    TranEncod = TransactionEncoder()
    te_ary = TranEncod.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=TranEncod.columns_)

.. code:: ipython3

    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Coffee</th>
          <th>Diaper</th>
          <th>Drink</th>
          <th>Eggs</th>
          <th>Milk</th>
          <th>Nuts</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>False</td>
          <td>True</td>
          <td>True</td>
          <td>False</td>
          <td>False</td>
          <td>True</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
          <td>True</td>
          <td>True</td>
          <td>False</td>
          <td>False</td>
          <td>False</td>
        </tr>
        <tr>
          <th>2</th>
          <td>False</td>
          <td>True</td>
          <td>True</td>
          <td>True</td>
          <td>False</td>
          <td>False</td>
        </tr>
        <tr>
          <th>3</th>
          <td>False</td>
          <td>False</td>
          <td>False</td>
          <td>True</td>
          <td>True</td>
          <td>True</td>
        </tr>
        <tr>
          <th>4</th>
          <td>True</td>
          <td>True</td>
          <td>False</td>
          <td>True</td>
          <td>True</td>
          <td>True</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    frequent_itemsets=apriori(df,min_support=0.5,use_colnames=True)
    frequent_itemsets['length']=frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>support</th>
          <th>itemsets</th>
          <th>length</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.8</td>
          <td>(Diaper)</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.6</td>
          <td>(Drink)</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.6</td>
          <td>(Eggs)</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.6</td>
          <td>(Nuts)</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.6</td>
          <td>(Drink, Diaper)</td>
          <td>2</td>
        </tr>
      </tbody>
    </table>
    </div>




.. code:: ipython3

    apriori(df,min_support=0.6, use_colnames=True)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>support</th>
          <th>itemsets</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.8</td>
          <td>(Diaper)</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.6</td>
          <td>(Drink)</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.6</td>
          <td>(Eggs)</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.6</td>
          <td>(Nuts)</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.6</td>
          <td>(Drink, Diaper)</td>
        </tr>
      </tbody>
    </table>
    </div>




.. code:: ipython3

    frequent_itemsets[(frequent_itemsets['length']==2)&
                    ( frequent_itemsets['support']>=0.5) ]




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>support</th>
          <th>itemsets</th>
          <th>length</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>4</th>
          <td>0.6</td>
          <td>(Drink, Diaper)</td>
          <td>2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    
    
    frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>support</th>
          <th>itemsets</th>
          <th>length</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.8</td>
          <td>(Diaper)</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.6</td>
          <td>(Drink)</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.6</td>
          <td>(Eggs)</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.6</td>
          <td>(Nuts)</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.6</td>
          <td>(Drink, Diaper)</td>
          <td>2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    
    frequent_itemsets[ frequent_itemsets['itemsets'] == {'Diaper', 'Drink'} ]




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>support</th>
          <th>itemsets</th>
          <th>length</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>4</th>
          <td>0.6</td>
          <td>(Drink, Diaper)</td>
          <td>2</td>
        </tr>
      </tbody>
    </table>
    </div>




