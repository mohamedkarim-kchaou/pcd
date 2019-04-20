import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import style
import pylab as plt
import csv
from django.conf import settings


def afficher_annees_precedentes(csv_file, annee):
    sns.set()
    style.use('ggplot')

    regions = []
    flu_young = []
    sari_young = []
    cons_young = []
    flu_ad = []
    sari_ad = []
    cons_ad = []
    flu_ag = []
    sari_ag = []
    cons_ag = []
    tot_flu = []
    tot_sari = []
    tot_cons = []
    t_flu = []
    t_sari = []
    dossier = settings.STATIC_URL+"main/images/resultats_annees_precedentes"
    with open(csv_file) as csvfile:
        ilireader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(ilireader, None)  # skipping the header
        for row in ilireader:
            regions.append(row[0])
            flu_young.append(int(row[1]))
            sari_young.append(int(row[2]))
            cons_young.append(int(row[3]))
            flu_ad.append(int(row[4]))
            sari_ad.append(int(row[5]))
            cons_ad.append(int(row[6]))
            flu_ag.append(int(row[7]))
            sari_ag.append(int(row[8]))
            cons_ag.append(int(row[9]))
            tot_flu.append(int(row[10]))
            tot_cons.append(int(row[11]))
            tot_sari.append(int(row[12]))
            t_flu.append(float(row[13]))
            t_sari.append(float(row[14]))

    y_pos = np.arange(len(regions))
    plt.rcParams['figure.figsize'] = (8, 6)

    plt.subplots()
    plt.bar(regions, tot_flu, label='Total Flu', color='lightcoral', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Total flu')
    plt.title('Total flu per region in Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/1totflu.png')

    plt.subplots()
    plt.bar(regions, tot_sari, label='Total Sari', color='lightskyblue', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Total sari')
    plt.title('Total sari per region in Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/2totsari.png')

    plt.subplots()
    plt.bar(regions, tot_cons, label='Total Consultant', color='yellowgreen', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Total consultants')
    plt.title('Total consultants per region in Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/3totcons.png')

    plt.subplots()
    plt.bar(regions, flu_young, label='< 5 years', color='gold', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Flu')
    plt.title('Influenza cases for people younger than 5 years in regions of Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/4fluyo09-10.png')

    plt.subplots()
    plt.bar(regions, flu_ad, label='5 - 16 years', color='magenta', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Flu')
    plt.title('Influenza cases for people between 5 and 16 years in regions of Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/5fluad.png')

    plt.subplots()
    plt.bar(regions, flu_ag, label='> 16 years', color='green', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Flu')
    plt.title('Influenza cases for people older than 16 years in regions of Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/6fluag.png')

    plt.subplots()
    plt.bar(regions, sari_young, label='< 5 years', color='gold', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Sari')
    plt.title('Sari cases for people < 5 years and governorates in Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/7sariyo.png')

    plt.subplots()
    plt.bar(regions, sari_ad, label='5 - 16 years', color='magenta', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Sari')
    plt.title('Sari cases for people between 5 and 16 years in regions of Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/8sariad.png')

    plt.subplots()
    plt.bar(regions, sari_ag, label='> 16 years', color='green', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Sari')
    plt.title('Sari cases for people older than 16 years in regions of Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/9sariag.png')

    plt.subplots()
    plt.bar(regions, t_flu, label='Flu rate', color='lightcoral', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Flu rate')
    plt.title('Flu rate per region in Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/9ztflu.png')

    plt.subplots()
    plt.bar(regions, t_sari, label='Sari rate', color='lightskyblue', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Region')
    plt.ylabel('Sari rate')
    plt.title('Sari rate per region in Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/9zztsari.png')

    for val in t_flu:
        if val >= 10.0:
            ind = t_flu.index(val)
            print(regions[ind], 'has an epidemiologic year')

