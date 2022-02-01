import sqlite3
from dbfread import DBF
def get_fields(table):
    """get the fields and sqlite types for a dbf table"""
    typemap = {
        "F": "FLOAT",
        "L": "BOOLEAN",
        "I": "INTEGER",
        "C": "TEXT",
        "N": "REAL",  # because it can be integer or float
        "M": "TEXT",
        "D": "DATE",
        "T": "DATETIME",
        "0": "INTEGER",
    }

    fields = {}
    for f in table.fields:
        fields[f.name] = typemap.get(f.type, "TEXT")
    return fields


def create_table_statement(table_name, fields):
    defs = ", ".join(['"%s" %s' % (fname, ftype) for (fname, ftype) in fields.items()])
    sql = 'create table "%s" (%s)' % (table_name, defs)
    return sql


def insert_table_statement(table_name, fields):
    refs = ", ".join([":" + f for f in fields.keys()])
    sql = 'insert into "%s" values (%s)' % (table_name, refs)
    return sql


def copy_table(cursor, table):
    """Add a dbase table to an open sqlite database"""
    cursor.execute("drop table if exists %s" % table.name)
    fields = get_fields(table)

    sql = create_table_statement(table.name, fields)
    cursor.execute(sql)

    sql = insert_table_statement(table.name, fields)

    for rec in table:
        cursor.execute(sql, list(rec.values()))


def main():
    output_file = "himalayan_database.sqlite"
    tables = ["exped", "members", "peaks", "refer"]
    conn = sqlite3.connect(output_file)
    cursor = conn.cursor()

    for table_name in tables:
        table_file = f"Himalayan Database\HIMDATA\{table_name}.DBF"
        dbf_table = DBF(table_file, lowernames=True, encoding=None, char_decode_errors="strict")
        copy_table(cursor, dbf_table)

    conn.commit()


if __name__ == "__main__":
    main()


#import of top mountains data to manipulate and create a visual of top 14 peak ascent data

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',60)

tmd = pd.read_csv('Top mountains data.csv')
print(tmd.head())
tmd.sort_values('Height_m', ascending =False)
tmd8000 = tmd[tmd['Height_m']>=8000]
tmd8000_ind = tmd8000.set_index('rank')
print('The top 14 highest peaks globally are over 8000m and are show below')
print('----------------------------------------------------------------------------')
print(tmd8000_ind[['Height_m','Mountain name(s)','Range', 'ascents_first']])
print(tmd8000_ind.isna().sum())
tmd8000_ind.rename(columns = {'Mountain name(s)': 'Mountain'}, inplace =True)
tmd8000_ind.loc[tmd8000_ind.Mountain == 'Mount Everest\nSagarmath\nChomolungma', 'Mountain'] = 'Mount Everest'
tmd8000_ind.loc[tmd8000_ind.Mountain == 'Gasherbrum I\nHidden Peak\nK5', 'Mountain'] = 'Gasherbrum I'
tmd8000_ind.loc[tmd8000_ind.Mountain == 'Gasherbrum II\nK4', 'Mountain'] = 'Gasherbrum II'
tmd8000_ind.loc[tmd8000_ind.Mountain == 'Shishapangma\nGosainthan', 'Mountain'] = 'Shishapangma'

tmd8000_ind.to_csv('tmd8000_ind.csv')

#Visual of the first ascents fo 14 highest peaks

import matplotlib.pyplot as plt
import seaborn as sns

print(plt.style.available)
plt.figure(figsize=(12, 8))
plt.style.use('bmh')
g = sns.scatterplot(x='Mountain', y='ascents_first', data=tmd8000_ind, hue='Height_m',s=200)
g.set_title('Peaks Over 8000m Ascent Data (All Mountains)', y=1.03)
plt.xlabel('Mountain Name', fontsize=14)
plt.ylabel('Year of First Ascent', fontsize=14)
plt.xticks(rotation=60, fontsize=12)
plt.rc('xtick', labelsize=8)
plt.subplots_adjust(bottom=0.26)

plt.savefig('Top14Ascent.png')
plt.show()

#import datasets

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from matplotlib import style

engine = create_engine('sqlite:///himalayan_database.sqlite')
table_names =engine.table_names()
print(table_names)

#pulling first table exped into a dataframe for inspection

con=engine.connect()
exped =pd.read_sql_query('SELECT * FROM exped', engine)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',10)
print(exped.head())
print(exped.columns)

exped =pd.read_sql_query('SELECT expid, peakid, year, season, nation, bcdate, smtdate, smttime, o2used FROM exped', engine)
print(exped['peakid'].unique())
shape1 = exped.shape
print(shape1)
print(exped.isnull().sum())

exped.to_csv('exped.csv')

#pulling second table peaks into a dataframe for inspection

peaks =pd.read_sql_query('SELECT * FROM peaks', engine)
print(peaks.head())
print(peaks.columns)

peaks =pd.read_sql_query('SELECT * FROM peaks WHERE heightm >= 8000', engine)
shape2 = peaks.shape
print(shape2)
print(peaks['peakid'].unique())
print(peaks['pkname'].unique())
print(peaks['heightm'].unique())
peak_hgt =peaks[['pkname', 'heightm']]
print(peak_hgt)

peaks['heightm'] =peaks['heightm'].astype(str)

#creating dictionary of current peak names and their height in metres

peaks_dict = {'Annapurna I': '8091', 'Annapurna I East': '8026', 'Annapurna I Middle': '8051', 'Cho Oyu': '8188',\
              'Dhaulagiri I': '8167','Everest': '8849','Kangchenjunga Central': '8473', 'Kangchenjunga':'8586',\
              'Kangchenjunga South':'8476','Lhotse':'8516','Lhotse Shar': '8382', 'Makalu':'8485', 'Manaslu': '8163',\
              'Yalung Kang':'8505', 'Lhotse Middle':'8410','Yalung Kang West':'8077'}

for key, value in peaks_dict.items():
    print('The height of ' +str(key) + ' is ' + str(value))

#creating dictionary to convert peak heights to match heighest point of mountain

peaks_hgt_dict = {'8586.0':'8586','8516.0': '8516','8849.0':'8849','8188.0': '8188','8167.0':'8167',\
                  '8091.0': '8091','8026.0': '8091', '8051.0': '8091','8473.0': '8586','8476.0':'8586',\
                  '8505.0':'8586', '8077.0':'8586','8382.0':'8516','8410.0': '8516','8485.0':'8485','8163.0':'8613'}

peaks_2 = peaks.replace({'heightm': peaks_hgt_dict})

#creating dictionary to convert peak names to name of main mountain e.g. Kangchenjunga

pkname_dict = {'Annapurna I': 'Annapurna', 'Annapurna I East': 'Annapurna', 'Annapurna I Middle': 'Annapurna',\
               'Kangchenjunga Central':'Kangchenjunga', 'Kangchenjunga South':'Kangchenjunga', 'Yalung Kang':'Kangchenjunga',\
               'Yalung Kang West':'Kangchenjunga', 'Lhotse Shar':'Lhotse', 'Lhotse Middle': 'Lhotse'}

peaks_3 = peaks_2.replace({'pkname': pkname_dict})
print(peaks_3[['pkname', 'heightm']])
print(peaks_3['pkname'].unique())
print(peaks_3.isnull().sum())

peaks_3.to_csv('peaks_3.csv')

#joining both of these dataframes by peakid

exped_peaks= exped.merge(peaks_3, on='peakid', suffixes = ('_exped', '_peaks'))
print(exped_peaks.head())
print(exped_peaks.columns)
shape5 = exped_peaks.shape
print(shape5)

exped_peaks.to_csv('exped_peaks.csv')

#pulling third table members into a dataframe for inspection
members = pd.read_sql_query('SELECT * FROM members', engine)
print(members.head())
print(members.columns)
shape4 = members.shape
print(shape4)

members.to_csv('members.csv')

#To filter and pull certain columns from members table

members_f = pd.read_sql_query('SELECT expid, membid, peakid, myear ,mseason, fname, lname, sex, yob,\
                              calcage, citizen, occupation, bconly, nottobc, disabled, msuccess, msmtdate1, msmttime1, mo2used,\
                              mo2none, death, deathdate FROM members', engine)
print(members_f.head())
print(members_f.columns)
shape4 = members_f.shape
print(shape4)

members_f.to_csv('members_f.csv')

#joining all 3 tables, exped, peaks and members

exped_peaks_members = exped_peaks.merge(members, on='expid', suffixes = ('_exped', '_members'))
print(exped_peaks_members.head())
print(exped_peaks_members.columns)
shape5 = exped_peaks_members.shape
print(shape5)

exped_peaks_members.to_csv('exped_peaks_members.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',60)

#Filling missing values and replacing binary values and column names

print(exped_peaks_members.isnull().sum())
exped_peaks_members['bcdate'] = exped_peaks_members['bcdate'].fillna(0)
exped_peaks_members['smtdate'] = exped_peaks_members['smtdate'].fillna(0)
exped_peaks_members['msmtdate1'] = exped_peaks_members['msmtdate1'].fillna(0)
exped_peaks_members['o2used'] = exped_peaks_members['o2used'].replace([0, 1], ['No', 'Yes'])
exped_peaks_members['bconly'] = exped_peaks_members['bconly'].replace([0, 1], ['No', 'Yes'])
exped_peaks_members['msuccess'] = exped_peaks_members['msuccess'].replace([0, 1], ['No', 'Yes'])
exped_peaks_members['mo2used'] = exped_peaks_members['mo2used'].replace([0, 1], ['No', 'Yes'])
exped_peaks_members['mo2none'] = exped_peaks_members['mo2none'].replace([0, 1], ['No', 'Yes'])
exped_peaks_members['death'] = exped_peaks_members['death'].replace([0, 1], ['No', 'Yes'])
exped_peaks_members['deathdate'] = exped_peaks_members['deathdate'].fillna(0)

exped_peaks_members.rename(columns = {'mo2used': 'Oxygen Used'}, inplace =True)
exped_peaks_members.rename(columns = {'msuccess': 'Summit'}, inplace =True)

exped_peaks_members.to_csv('exped_peaks_members.csv')

epm1 =exped_peaks_members[['expid', 'year', 'season', 'nation', 'heightm', 'open', 'bcdate', 'myear','fname', 'lname', 'sex', 'calcage', 'smtdate',\
                           'smttime','o2used','pkname', 'bconly', 'nottobc', 'Summit', 'msmtdate1', 'msmttime1',\
                           'Oxygen Used', 'mo2none','death', 'deathdate']]

print(epm1.head())

epm1.to_csv('emp1.csv')


#Summary Statistics
print('The average age of all climbers of the Nepal Himal 8000m peaks is:')
print(epm1['calcage'].mean().round(2))
print('The first year on record for attempted 8000m ascents in Nepal Himalayas is:')
print(epm1['myear'].min())
print('Below are the names and peak name for the 1905 expidition (where known)')
first = (epm1['myear'] =='1905')
first_att = epm1[first]
print(first_att[['myear','pkname', 'fname', 'lname']])
print('The total number of summit attempts at Nepal Himalaya on record is:' )
att = epm1['open'].sum()
print(att)
print('The split of total recorded climbers by gender is:')
gen=epm1['sex'].value_counts()
print(gen)



#Having a look at the first years of ascent for each peak
success = epm1['Summit'] =='Yes'
psuccess = epm1[success]

first_ascent =psuccess.groupby('pkname')['myear'].min()
print(first_ascent)

Anna = psuccess[(psuccess['myear'] == '1950') & (psuccess['pkname']== 'Annapurna')]
Anna_First = Anna[['pkname', 'fname','lname','myear']]
print('The details for the first climbers to summit Annapurna are shown below')
print('----------------------------------------------------------------------')
print(Anna_First)

Cho = psuccess[(psuccess['myear'] == '1954') & (psuccess['pkname'] == 'Cho Oyu')]
Cho_First = Cho[['pkname', 'fname','lname','myear']]
print('The details for the first climbers to summit Cho Oyu are shown below')
print('--------------------------------------------------------------------')
print(Cho_First)

Dhaul = psuccess[(psuccess['myear'] == '1960') & (psuccess['pkname'] == 'Dhaulagiri I ')]
Dhaul_First = Dhaul[['pkname', 'fname','lname','myear']]
print('The details for the first climbers to summit Dhaulagiri I are shown below')
print('-------------------------------------------------------------------------')
print(Dhaul_First)

Everest = psuccess[(psuccess['myear'] == '1953') & (psuccess['pkname'] == 'Everest')]
Everest_First = Everest[['pkname', 'fname','lname','myear']]
print('The details for the first climbers to summit Everest are shown below')
print('--------------------------------------------------------------------')
print(Everest_First)

Kang = psuccess[(psuccess['myear'] == '1955') & (psuccess['pkname'] == 'Kangchenjunga')]
Kang_First = Kang[['pkname', 'fname','lname','myear']]
print('The details for the first climbers to summit Kangchenjunga  are shown below')
print('---------------------------------------------------------------------------')
print(Kang_First)

Lhotse = psuccess[(psuccess['myear'] == '1956') & (psuccess['pkname'] == 'Lhotse')]
Lhotse_First = Lhotse[['pkname', 'fname','lname','myear']]
print('The details for the first climbers to summit Lhotse  are shown below')
print('--------------------------------------------------------------------')
print(Lhotse_First)

Mak = psuccess[(psuccess['myear'] == '1955') & (psuccess['pkname'] == 'Makalu')]
Mak_First = Mak[['pkname', 'fname','lname','myear']]
print('The details for the first climbers to summit Makalu are shown below')
print('-------------------------------------------------------------------')
print(Mak_First)

Man = psuccess[(psuccess['myear'] == '1956') & (psuccess['pkname'] == 'Manaslu')]
Man_First = Man[['pkname', 'fname','lname','myear']]
print('The details for the first climbers to summit Manaslu are shown below')
print('--------------------------------------------------------------------')
print(Man_First)

#Proportion of successful climbing attempts per peak

print('The split of successful and unsuccessful summits per Peak is shown below')
pk_sum = epm1.groupby(['pkname', 'Summit'])['open'].count()
print(pk_sum)

anna_prop = 397/ (1445+397) *100
anna_p =round(anna_prop, 2)
print('The proportion of all climbers (per attempt) to reach the summit of Annapurna is:-')
print(anna_p)

cho_prop = 3900/ (5074+3900) *100
cho_p =round(cho_prop, 2)
print('The proportion of all climbers (per attempt) to reach the summit of Cho Oyu is:-')
print(cho_p)

dhau_prop = 558/ (2128+558) *100
dhau_p =round(dhau_prop, 2)
print('The proportion of all climbers (per attempt) to reach the summit of Dhaulagiri I is:-')
print(dhau_p)

ever_prop = 10538/ (12107+10538) *100
ever_p =round(ever_prop, 2)
print('The proportion of all climbers (per attempt) to reach the summit of Everest is:-')
print(ever_p)

kang_prop = 549/ (1167+549) *100
kang_p =round(kang_prop, 2)
print('The proportion of all climbers (per attempt) to reach the summit of Kangchenjunga is:-')
print(kang_p)

lho_prop = 963/ (1855+963) *100
lho_p =round(lho_prop, 2)
print('The proportion of all climbers (per attempt) to reach the summit of Lhotse is:-')
print(lho_p)

mak_prop = 566/ (1874+566) *100
mak_p =round(mak_prop, 2)
print('The proportion of all climbers (per attempt) to reach the summit of Makalu is:-')
print(mak_p)

man_prop = 2967/ (2143+2967) *100
man_p =round(man_prop, 2)
print('The proportion of all climbers (per attempt) to reach the summit of Manaslu is:-')
print(man_p)


#creating variables based on conditions to filter data ranges and base camp/summit success, all 8000 peaks

epm_bc5070  =epm1[(epm1['bconly']== 'Yes') & (epm1['myear'] >='1950') & (epm1['myear'] <= '1970)')]
print(epm_bc5070)
epm_bc7190  =epm1[(epm1['bconly']== 'Yes') & (epm1['myear'] >='1971') & (epm1['myear'] <= '1990')]
epm_bc9105 = epm1[(epm1['bconly']== 'Yes') & (epm1['myear'] >='1991') & (epm1['myear'] <= '2005')]
epm_bc0621 = epm1[(epm1['bconly']== 'Yes') & (epm1['myear'] >='2006') & (epm1['myear'] <= '2021')]

epm_smt5070  =epm1[(epm1['Summit'] == 'Yes') & (epm1['myear'] >='1950') & (epm1['myear'] <= '1970')]
epm_smt7190  =epm1[(epm1['Summit'] == 'Yes') & (epm1['myear'] >='1971') & (epm1['myear'] <= '1990')]
epm_smt9105  =epm1[(epm1['Summit'] == 'Yes') & (epm1['myear'] >='1991') & (epm1['myear'] <= '2005')]
epm_smt0621  =epm1[(epm1['Summit'] == 'Yes') & (epm1['myear'] >='2006') & (epm1['myear'] <= '2021')]

bc1 =epm_bc5070.groupby('myear', as_index=False)['bconly'].count()
bc2 =epm_bc7190.groupby('myear', as_index=False)['bconly'].count()
bc3 =epm_bc9105.groupby('myear', as_index=False)['bconly'].count()
bc4 =epm_bc0621.groupby('myear', as_index=False)['bconly'].count()

smt1 =epm_smt5070.groupby('myear', as_index=False)['Summit'].count()
smt2 =epm_smt7190.groupby('myear', as_index=False)['Summit'].count()
smt3 =epm_smt9105.groupby('myear', as_index=False)['Summit'].count()
smt4 =epm_smt0621.groupby('myear', as_index=False)['Summit'].count()

#Plotting subplots of base camp and summit success

fig, ax =plt.subplots(4,1)

fig.set_size_inches([10,8])
ax[0].plot(bc1['myear'], bc1['bconly'], color='r')
ax[1].plot(bc2['myear'], bc2['bconly'])
ax[2].plot(bc3['myear'], bc3['bconly'], color='g')
ax[3].plot(bc4['myear'], bc4['bconly'], color= 'm')
fig.suptitle('Succesfull Climbers to Base Camp All Nepal Himal 8000 Peaks', fontsize=12)
fig.savefig('BC_Comparisons.png')
plt.show()

fig, ax =plt.subplots(4,1)

fig.set_size_inches([10,8])
ax[0].plot(smt1['myear'], smt1['Summit'], color='r')
ax[1].plot(smt2['myear'], smt2['Summit'])
ax[2].plot(smt3['myear'], smt3['Summit'], color='g')
ax[3].plot(smt4['myear'], smt4['Summit'], color= 'm')
fig.suptitle('Succesfull Climbers to Summit All Nepal Himal 8000 Peaks', fontsize=12)
fig.savefig('SMT_Comparisons.png')
plt.show()

#Creating varables for base camp and summit success to allow for split by peak name
bc5070 = epm_bc5070.groupby(['myear','pkname'],as_index=False)['bconly'].count()
print(bc5070)
bc7190 = epm_bc7190.groupby(['myear','pkname'],as_index=False)['bconly'].count()
bc9105 = epm_bc9105.groupby(['myear','pkname'],as_index=False)['bconly'].count()
bc0621 = epm_bc0621.groupby(['myear','pkname'],as_index=False)['bconly'].count()
smt5070 = epm_smt5070.groupby(['myear','pkname'],as_index=False)['Summit'].count()
smt7190 = epm_smt7190.groupby(['myear','pkname'],as_index=False)['Summit'].count()
smt9105 = epm_smt9105.groupby(['myear','pkname'],as_index=False)['Summit'].count()
smt0621 = epm_smt0621.groupby(['myear','pkname'],as_index=False)['Summit'].count()

order =['Annaapurne','Cho Oyu','Dhaulagiri','Everest','Kanchenjunga','Lhotse','Makalu','Manaslu']

plt.style.use('bmh')
plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(x='myear', y='bconly', data=bc5070, hue='pkname', hue_order=order)
g.set_title('Climbers to Base Camp Only 1950 to 1970', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize=14)
g.set_ylabel('Climbers to Base Camp Only', fontsize=14)
g.legend(title='Peak Name', title_fontsize='12', loc='upper left')
plt.savefig('bc5070.png')
plt.show()

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(x='myear', y='bconly', data=bc7190, hue='pkname',hue_order=order)
g.set_title('Climbers to Base Camp Only 1971 to 1990', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize=14)
g.set_ylabel('Climbers to Base Camp Only', fontsize=14)
g.legend(title='Peak Name', title_fontsize='12', loc='upper left')
plt.savefig('bc7190.png')
plt.show()


plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(x='myear', y='bconly', data=bc9105, hue='pkname', hue_order=order)
g.set_title('Climbers to Base Camp Only 1991 to 2005', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize = 14)
g.set_ylabel('Climbers to Base Camp Only', fontsize=14)
g.legend(title='Peak Name', title_fontsize='12', loc='upper left')
plt.savefig('bc9105.png')
plt.show()

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(x='myear', y='bconly', data=bc0621, hue='pkname', hue_order=order)
g.set_title('Climbers to Base Camp Only 2006 to 2021', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize=14)
g.set_ylabel('Climbers to Base Camp Only', fontsize=14)
g.legend(title='Peak Name', title_fontsize='12', loc='upper left')
plt.savefig('bc0621.png')
plt.show()

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
plt.ylim(0,10)
g = sns.barplot(x='myear', y='Summit', data=smt5070, hue='pkname',hue_order=order)
g.set_title('Climbers to Summit 1950 to 1970', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize=14)
g.set_ylabel('Climbers to Summit', fontsize=14)
g.legend(title='Peak Name', title_fontsize='12', loc='upper left')
plt.savefig('smt5070.png')
plt.show()


plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(x='myear', y='Summit', data=smt7190, hue='pkname',hue_order=order)
g.set_title('Climbers to Summit 1971 to 1990', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize=14)
g.set_ylabel('Climbers to Summit', fontsize=14)
g.legend(title='Peak Name', title_fontsize='12', loc='upper left')
plt.savefig('smt7190.png')
plt.show()

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(x='myear', y='Summit', data=smt9105, hue='pkname',hue_order=order)
g.set_title('Climbers to Summit 1991 to 2005', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize=14)
g.set_ylabel('Climbers to Summit', fontsize=14)
g.legend(title='Peak Name', title_fontsize='12', loc='upper left')
plt.savefig('smt9105.png')
plt.show()

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(x='myear', y='Summit', data=smt0621, hue='pkname',hue_order=order)
g.set_title('Climbers to Summit 2006 to 2021', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize=14)
g.set_ylabel('Climbers to Summit', fontsize=14)
g.legend(title='Peak Name', title_fontsize='12', loc='upper left')
plt.savefig('smt0621.png')
plt.show()

#Variables to allow grouping by whether oxygen was used and gender

o2used5070 = epm_smt5070.groupby(['myear','Oxygen Used'],as_index=False)['Summit'].count()
print(smt5070)
o2used7190 = epm_smt7190.groupby(['myear','sex', 'Oxygen Used'],as_index=False)['Summit'].count()
o2used9105 = epm_smt9105.groupby(['myear','sex', 'Oxygen Used'],as_index=False)['Summit'].count()
o2used0621 = epm_smt0621.groupby(['myear','sex', 'Oxygen Used'],as_index=False)['Summit'].count()

plt.style.use('ggplot')
plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(data=o2used5070, x='myear', y='Summit', hue='Oxygen Used')
g.set_title('Climbers to Summit All Nepal Himal 8000 Peaks 1950 to 1970', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize=14)
g.set_ylabel('Climbers to Summit', fontsize=14)
plt.savefig('o25070.png')
plt.show()

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(x='myear', y='Summit', data =o2used7190, hue='Oxygen Used', ci=None)
g.set_title('Climbers to Summit All Nepal 8000 peaks 1971 to 1990', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize = 14)
g.set_ylabel('Climbers to Summit', fontsize = 14)
g.legend(title='Oxygen Used', title_fontsize='12', loc='upper left')
plt.savefig('o27190')
plt.show()

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(x='myear', y='Summit', data=o2used9105, hue='Oxygen Used', ci=None)
g.set_title('Climbers to Summit - All Nepal 8000 peaks 1991 to 2005', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize=14)
g.set_ylabel('Climbers to Summit', fontsize=14)
g.legend(title='Oxygen Used', title_fontsize='12', loc='upper left')
plt.savefig('o29105')
plt.show()

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.barplot(x='myear', y='Summit', data =o2used0621, hue='Oxygen Used', ci=None)
g.set_title('Climbers to Summit - All Nepal 8000 peaks 2006 to 2021', fontsize=14,  y=1.03)
g.set_xlabel('Year', fontsize = 14)
g.set_ylabel('Climbers to Summit', fontsize=14)
g.legend(title='Oxygen Used', title_fontsize='12', loc='upper left')
plt.savefig('o20621')
plt.show()

plt.style.use('seaborn-bright')
g= sns.relplot(x='myear', y ='Summit', data=o2used0621, kind='line', hue='Oxygen Used',col ='sex')
g.fig.set_figwidth(16)
g.fig.set_figheight(8)
g.set_axis_labels('Year', 'Successful Ascents', fontsize=14)
plt.savefig('o2gender.png')
plt.show()

#Visuals for all recorded climbing attempts, summit success and oxygen used
plt.style.use('seaborn')
plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.histplot(data=epm1, x='pkname', multiple='stack', hue='Summit')
g.set_title('Climbing Activity All Nepal Himal 8000 Peaks (On Record)', fontsize=14,  y=1.03)
g.set_xlabel('Peak Name', fontsize=14)
g.set_ylabel('Number of Climbing Attempts', fontsize=14)
plt.savefig('smtall.png')
plt.show()

plt.style.use('seaborn')
plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g = sns.histplot(data=epm1, x='pkname', multiple='stack', hue='Oxygen Used')
g.set_title('Climbing Activity For All Nepal Himal 8000 Peaks (On Record)', fontsize=14,  y=1.03)
g.set_xlabel('Peak Name', fontsize = 14)
g.set_ylabel('Number of Climbing Attempts', fontsize=14)
plt.savefig('o2all.png')
plt.show()

#Looking af total climbing activity and death numbers. The 'open' column refers to total numbers
death_all = epm1['death'] == 'Yes'
death_total = epm1[death_all]

death = death_total.groupby(['pkname','Summit'],as_index=False)['death'].count()
print(death)

total = epm1.groupby(['pkname','Summit'],as_index=False)['open'].sum()
print(total)
total_d = epm1.groupby(['pkname','death'],as_index=False)['open'].sum()
print(total_d)

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g= sns.barplot(x='pkname', y='open', data=total, hue='Summit', ci=None)
g.set_title('Climbing Activity (All Years)', fontsize=14,  y=1.03)
g.set_xlabel('Peak Name', fontsize = 14)
g.set_ylabel('Number of Climbing Attempts', fontsize=14)
g.legend(title='Summit Reached', title_fontsize='12', loc='upper left')
plt.savefig('tot_climbers.png')
plt.show()

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=10)
g= sns.barplot(x='pkname', y='death', data= death,hue='Summit', ci=None)
g.set_title('Total Number of Deaths Recorded (All Years)', fontsize=14,  y=1.03)
g.set_xlabel('Peak Name', fontsize=14)
g.set_ylabel('Number of Deaths', fontsize=14)
g.legend(title='Summit Reached', title_fontsize='12', loc='upper left')
plt.savefig('death.png')
plt.show()

#Final calculations on proportion of deaths per peak
anna_death = 76/ (1766+76) *100
anna_d =round(anna_death,2)
print('The proportion of deaths V climbing activity on Annapurna is:-')
print(anna_d)

cho_death = 52/ (8922+52) *100
cho_d =round(cho_death,2)
print('The proportion of deaths V climbing activity on Cho Oyu is:-')
print(cho_d)

dhau_death = 85/ (2601+85) *100
dhau_d =round(dhau_death,2)
print('The proportion of deaths V climbing activity on Dhaulagiri I is:-')
print(dhau_d)

ever_death = 312/ (22333+312) *100
ever_d =round(ever_death,2)
print('The proportion of deaths V climbing activity on Everest is:-')
print(ever_d)

kang_death = 61/ (1655+61) *100
kang_d =round(kang_death,2)
print('The proportion of deaths V climbing activity on Kangchenjunga is:-')
print(kang_d)

lho_death = 31/ (2787+31) *100
lho_d =round(lho_death,2)
print('The proportion of deaths V climbing activity on Lhotse is:-')
print(lho_d)

mak_death = 48/ (2392+48) *100
mak_d =round(mak_death,2)
print('The proportion of deaths V climbing activity on Makalu is:-')
print(mak_d)

man_death = 86/ (5024+86) *100
man_d =round(man_death,2)
print('The proportion of deaths V climbing activity on Manaslu is:-')
print(man_d)

print('This makes Annapurna the deadliest mountain of the 8 Nepal Himalayan 8000m peaks')


