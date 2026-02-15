"""
Pipeline de preprocessing pour les données immobilières
Ce module contient toutes les fonctions de nettoyage et de transformation des données
"""

import pandas as pd
import numpy as np
import re
from typing import Optional


def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV
    
    Args:
        filepath: Chemin vers le fichier CSV
        
    Returns:
        DataFrame pandas avec les données chargées
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes non nécessaires pour l'analyse
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame sans les colonnes supprimées
    """
    columns_to_drop = ["scraped_at", "url", "bathrooms"]
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols)
    return df


def remove_missing_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes sans prix (variable cible)
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame sans les lignes avec prix manquant
    """
    df = df.dropna(subset=['price'])
    return df


def extract_rooms_from_title(title: str) -> float:
    """
    Extrait le nombre de pièces depuis le titre (ex: S1, S2, S3...)
    
    Args:
        title: Titre de l'annonce
        
    Returns:
        Nombre de pièces ou NaN si non trouvé
    """
    if pd.isna(title):
        return np.nan
    # Recherche S suivi d'un chiffre (ex: S1, S2, s+1)
    match = re.search(r'[sS]\+?(\d+)', str(title))
    if match:
        return float(match.group(1))
    return np.nan


def update_features_from_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Met à jour les caractéristiques binaires basées sur les mots-clés dans le titre
    et extrait le nombre de pièces si manquant
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame avec les caractéristiques mises à jour
    """
    # Extraction du nombre de pièces depuis le titre pour les valeurs manquantes
    mask_rooms = df['rooms'].isna()
    df.loc[mask_rooms, 'rooms'] = df.loc[mask_rooms, 'title'].apply(extract_rooms_from_title)
    
    # Mapping des caractéristiques et leurs mots-clés
    features_map = {
        'swimming_pool': ['piscine', 'pool'],
        'garden': ['jardin', 'garden'],
        'terrace': ['terrasse', 'terrace'],
        'garage': ['garage', 'parking', 'sous-sol'],
        'elevator': ['ascenseur'],
        'air_conditioning': ['climatisation', 'clim'],
        'heating': ['chauffage'],
        'equipped_kitchen': ['cuisine équipée', 'cuisine installée'],
        'security': ['sécurité', 'gardien', 'surveillance']
    }
    
    for col, keywords in features_map.items():
        if col in df.columns:
            pattern = '|'.join(keywords)
            # Met à jour la colonne à 1 si un mot-clé est trouvé dans le titre
            df.loc[df['title'].str.contains(pattern, case=False, na=False), col] = 1
    
    return df


def remove_residence_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes de type résidence sans informations suffisantes
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame sans les lignes problématiques
    """
    pattern = r'(?i)\brésidence\b'
    mask_residence = (df['property_type'].isna() & 
                     df['title'].str.contains(pattern, na=False) & 
                     df['rooms'].isna())
    
    df = df[~mask_residence]
    print(f"Supprimé {mask_residence.sum()} lignes de type résidence sans infos")
    return df


def extract_property_type_from_title(title: str, property_keywords: dict) -> str:
    """
    Extrait le type de propriété depuis le titre
    
    Args:
        title: Titre de l'annonce
        property_keywords: Dictionnaire des types de propriété et leurs mots-clés
        
    Returns:
        Type de propriété ou NaN si non trouvé
    """
    if pd.isna(title):
        return np.nan
    title_lower = str(title).lower()
    for prop_type, keywords in property_keywords.items():
        for kw in keywords:
            if kw in title_lower:
                return prop_type
    return np.nan


def fill_property_type_from_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplit les types de propriété manquants en analysant le titre
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame avec les types de propriété complétés
    """
    property_keywords = {
        'Appartement': ['appartement', 'app', 's1', 's2', 's3', 's4', 's0', 's5',
                       's 0', 's 1', 's 2', 's 3', 's 4', 's 5'],
        'Villa': ['villa', 'pavillion', 'vila', 'rdc', 'rez de chaussée'],
        'Duplex': ['duplex'],
        'Maison': ['maison'],
        'Immeuble': ['immeuble'],
        'Terrain': ['terrain'],
        'Studio': ['studio'],   
        'Triplex': ['triplex'],
        'Panthouse': ['panthouse', 'penthouse'],
        'Résidence': ['résidence'],
        'Chalet': ['chalet']
    }
    
    mask_prop = df['property_type'].isna()
    df.loc[mask_prop, 'property_type'] = df.loc[mask_prop, 'title'].apply(
        lambda x: extract_property_type_from_title(x, property_keywords)
    )
    
    print(f"Valeurs manquantes de property_type restantes: {df['property_type'].isna().sum()}")
    return df


def remove_missing_property_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes sans type de propriété
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame sans les lignes avec property_type manquant
    """
    df = df.dropna(subset=['property_type'])
    return df


def remove_all_missing_key_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes où toutes les caractéristiques clés sont manquantes
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame nettoyé
    """
    df = df.dropna(subset=['rooms', 'area_m2'], how='all')
    df = df.dropna(subset=['rooms', 'property_type'], how='all')
    return df


def fill_rooms_by_area(df: pd.DataFrame, row: pd.Series) -> float:
    """
    Remplit les pièces manquantes basé sur la surface et le type de propriété
    
    Args:
        df: DataFrame complet (pour trouver des valeurs similaires)
        row: Ligne à traiter
        
    Returns:
        Nombre de pièces (médiane des propriétés similaires) ou NaN
    """
    if pd.notna(row['rooms']):
        return row['rooms']
    
    prop_type = row['property_type']
    area = row['area_m2']
    
    if pd.isna(area):
        return np.nan
    
    # Trouve les lignes avec même type et surface similaire (±20m²)
    similar = df[(df['property_type'] == prop_type) & 
                 (df['area_m2'] >= area - 20) & 
                 (df['area_m2'] <= area + 20) &
                 (df['rooms'].notna())]
    
    if len(similar) > 0:
        return similar['rooms'].median()
    return np.nan


def fill_area_by_rooms(df: pd.DataFrame, row: pd.Series) -> float:
    """
    Remplit la surface manquante basée sur le nombre de pièces et le type
    
    Args:
        df: DataFrame complet (pour trouver des valeurs similaires)
        row: Ligne à traiter
        
    Returns:
        Surface (moyenne des propriétés similaires) ou NaN
    """
    if pd.notna(row['area_m2']):
        return row['area_m2']
    
    prop_type = row['property_type']
    rooms = row['rooms']
    
    if pd.isna(rooms):
        return np.nan
    
    # Trouve les lignes avec même type et même nombre de pièces
    similar = df[(df['property_type'] == prop_type) & 
                 (df['rooms'] == rooms) &
                 (df['area_m2'].notna())]
    
    if len(similar) > 0:
        return similar['area_m2'].mean()
    return np.nan


def impute_rooms_and_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes pour rooms et area_m2 basé sur des propriétés similaires
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame avec valeurs imputées
    """
    # Remplit rooms basé sur area et property_type
    df['rooms'] = df.apply(lambda row: fill_rooms_by_area(df, row), axis=1)
    
    # Remplit area_m2 basé sur rooms et property_type
    df['area_m2'] = df.apply(lambda row: fill_area_by_rooms(df, row), axis=1)
    
    print("Valeurs manquantes après imputation initiale:")
    print(df[['area_m2', 'rooms']].isna().sum())
    
    # Supprime les lignes avec rooms toujours manquant
    df = df.dropna(subset=['rooms'])
    
    return df


def map_region(location: str) -> str:
    """
    Mappe la localisation vers une région géographique
    
    Args:
        location: Localisation de la propriété
        
    Returns:
        Région géographique ('Grand Tunis', 'Sahel', 'North-East', ou 'South')
    """
    if pd.isna(location):
        return "South"
    loc = str(location).lower()

    # Mots-clés pour Grand Tunis
    grand_tunis_kw = [
        'tunis', 'ariana', 'ben arous', 'benarous', 'manouba',
        'la goulette', 'goulette', 'lac', 'berge du lac', 'les berges',
        'carthage', 'sidi bou said', 'sidi bou saïd', 'bardo',
        'el menzah', 'menzah', 'ennasr', 'nasr',
        'el aouina', 'aouina', 'kheireddine',
        'el omrane', 'omrane', 'el ouardia', 'ouardia',
        'kabaria', 'ibn sina', 'jbel lahmar',
        'el hrairia', 'hrairia',
        'sidi hassine', 'douar hicher',
        'ettahrir', 'intilaka',
        'raoued', 'ghazela', 'cité ghazela', 'borj louzir',
        'mnihla', 'soukra', 'la soukra',
        'bou mhel', 'boumhel', 'boumhel bassatine',
        'el mourouj', 'mourouj',
        'fouchana', 'borj cedria', 'hammam chatt',
        'rades', 'ezzahra', 'mornag', 'megrine', 'mégrine',
        'bir el bey',
        'manouba', 'den den', 'denden',
        'tebourba', 'jedaida', 'douar hicher',
        'borj el amri', 'mornaguia'
    ]

    sahel_kw = [
        'bizerte', 'bizert', 'metline', 'ras jebel', 'cap zebib',
        'menzel jemil', 'menzel bourguiba', 'el alia',
        'ghar el melh', 'utique', 'raf raf', 'zarzouna',
        'nabeul', 'hammamet', 'kelibia', 'kélibia',
        'korba', 'beni khiar', 'el maamoura',
        'menzel temime', 'lebna', 'haouaria', 'el haouaria',
        'tazarka', 'dar chaabane', 'dar châabane',
        'hammam el ghezaz', 'ghezèze', 'soliman',
        'beni khalled', 'bou argoub',
        'sousse', 'akouda', 'chott meriem',
        'kalaa kebira', 'kalâa kebira',
        'kalaa sghira', 'kalâa sghira',
        'sahloul', 'khezama', 'tantana',
        'hergla', 'bouficha', 'sidi abdelhamid',
        'sahline', 'barraket essahel',
        'monastir', 'ksar hellal', 'moknine',
        'bekalta', 'teboulba', 'ksibet el mediouni',
        'ksibet el-médiouni',
        'mahdia', 'salakta', 'ksour essef',
        'chebba', 'essouassi'
    ]
    
    north_east_kw = [
        'le kef', 'béja', 'beja', 'jendouba',
        'tabarka', 'ain draham'
    ]

    if any(kw in loc for kw in grand_tunis_kw):
        return "Grand Tunis"
    elif any(kw in loc for kw in sahel_kw):
        return "Sahel"
    elif any(kw in loc for kw in north_east_kw):
        return "North-East"
    else:
        return "South"


def add_region_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'Region' basée sur la localisation
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame avec la colonne Region ajoutée
    """
    df['Region'] = df['location'].apply(map_region)
    print("Distribution des annonces par région:")
    print(df['Region'].value_counts())
    return df


def final_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Effectue l'imputation finale pour area_m2 et rooms
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame avec imputation finale
    """
    # Calcul des médianes globales pour fallback
    global_area_median = df['area_m2'].median()
    global_rooms_median = df['rooms'].median()
    
    # Traitement spécial pour Terrain: rooms = 0
    mask_terrain_rooms = (df['property_type'] == 'Terrain') & (df['rooms'].isna())
    df.loc[mask_terrain_rooms, 'rooms'] = 0
    
    # Impute area_m2 et rooms en utilisant la médiane par property_type
    df['area_m2'] = df['area_m2'].fillna(
        df.groupby('property_type')['area_m2'].transform('median')
    )
    df['rooms'] = df['rooms'].fillna(
        df.groupby('property_type')['rooms'].transform('median')
    )
    
    # Fallback: utilise les médianes globales pour les valeurs restantes
    df['area_m2'] = df['area_m2'].fillna(global_area_median)
    df['rooms'] = df['rooms'].fillna(global_rooms_median)
    
    print("Valeurs manquantes après imputation finale:")
    print(df[['area_m2', 'rooms']].isna().sum())
    
    return df


def create_luxury_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une variable binaire 'is_luxury' basée sur les équipements
    
    Args:
        df: DataFrame d'entrée
        
    Returns:
        DataFrame avec la colonne is_luxury ajoutée
    """
    luxury_amenities = [
        'doorman', 'cellar', 'heating', 'security',
        'double_glazing', 'reinforced_door', 'equipped_kitchen', 'air_conditioning'
    ]
    
    # Compte le nombre d'équipements de luxe
    df['amenity_count'] = df[luxury_amenities].sum(axis=1)
    
    # Définit luxury: (Piscine == 1) OU (Nombre d'équipements >= 5)
    df['is_luxury'] = (
        (df['swimming_pool'] == 1) | (df['amenity_count'] >= 5)
    ).astype(int)
    
    # Supprime le compteur temporaire
    df = df.drop(columns=['amenity_count'])
    
    print("Classification luxury:")
    print(df['is_luxury'].value_counts())
    
    return df


def get_valid_property_types(df_train: pd.DataFrame) -> list:
    """
    Extrait les types de propriés valides depuis les données d'entraînement
    
    Args:
        df_train: DataFrame d'entraînement
        
    Returns:
        Liste des types de propriétés valides
    """
    return df_train['property_type'].unique().tolist()


def ensure_test_compatibility(df_test: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Assure la compatibilité du test avec le train:
    - Colonnes identiques
    - Pas de types de propriété non vus en entraînement
    
    Args:
        df_test: DataFrame de test
        df_train: DataFrame d'entraînement
        
    Returns:
        DataFrame de test compatible avec l'entraînement
    """
    # 1. Avoir les mêmes colonnes
    train_columns = set(df_train.columns)
    test_columns = set(df_test.columns)
    
    # Ajouter les colonnes manquantes avec 0
    for col in train_columns - test_columns:
        df_test[col] = 0
    
    # Supprimer les colonnes supplémentaires
    df_test = df_test[list(train_columns)]
    
    print(f"Compatibilité colonnes:")
    print(f"  - Colonnes d'entraînement: {len(train_columns)}")
    print(f"  - Colonnes de test: {len(df_test.columns)}")
    print(f"  - Identiques: {train_columns == test_columns}")
    
    # 2. Filtrer les types de propriété non vus
    valid_types = get_valid_property_types(df_train)
    unseen_types = df_test[~df_test['property_type'].isin(valid_types)]['property_type'].unique()
    
    if len(unseen_types) > 0:
        print(f"\nTypes de propriété non vus en entraînement: {unseen_types.tolist()}")
        initial_rows = len(df_test)
        df_test = df_test[df_test['property_type'].isin(valid_types)]
        removed = initial_rows - len(df_test)
        print(f"  - Lignes supprimées: {removed}")
        print(f"  - Lignes restantes: {len(df_test)}")
    else:
        print(f"\nTous les types de propriété du test sont valides ✓")
    
    return df_test


def preprocess_pipeline(input_filepath: str, output_filepath: str) -> pd.DataFrame:
    """
    Pipeline complet de preprocessing
    
    Args:
        input_filepath: Chemin vers le fichier CSV d'entrée
        output_filepath: Chemin vers le fichier CSV de sortie
        
    Returns:
        DataFrame nettoyé et traité
    """
    print("=" * 60)
    print("DÉMARRAGE DU PIPELINE DE PREPROCESSING")
    print("=" * 60)
    
    # 1. Chargement des données
    print("\n1. Chargement des données...")
    df = load_data(input_filepath)
    print(f"   Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # 2. Suppression des colonnes inutiles
    print("\n2. Suppression des colonnes inutiles...")
    df = drop_unnecessary_columns(df)
    print(f"   Colonnes restantes: {df.shape[1]}")
    
    # 3. Suppression des lignes sans prix
    print("\n3. Suppression des lignes sans prix...")
    initial_count = len(df)
    df = remove_missing_price(df)
    print(f"   Lignes supprimées: {initial_count - len(df)}")
    print(f"   Lignes restantes: {len(df)}")
    
    # 4. Extraction des features depuis le titre
    print("\n4. Extraction des features depuis le titre...")
    df = update_features_from_title(df)
    print(f"   Valeurs manquantes de 'rooms': {df['rooms'].isna().sum()}")
    
    # 5. Suppression des résidences problématiques
    print("\n5. Suppression des résidences sans infos...")
    df = remove_residence_rows(df)
    
    # 6. Remplissage du type de propriété
    print("\n6. Remplissage des types de propriété...")
    df = fill_property_type_from_title(df)
    
    # 7. Suppression des lignes sans type de propriété
    print("\n7. Suppression des lignes sans type de propriété...")
    initial_count = len(df)
    df = remove_missing_property_type(df)
    print(f"   Lignes supprimées: {initial_count - len(df)}")
    print(f"   Lignes restantes: {len(df)}")
    
    # 8. Suppression des lignes avec toutes les features clés manquantes
    print("\n8. Suppression des lignes sans features clés...")
    initial_count = len(df)
    df = remove_all_missing_key_features(df)
    print(f"   Lignes supprimées: {initial_count - len(df)}")
    print(f"   Lignes restantes: {len(df)}")
    
    # 9. Imputation des rooms et area basée sur similarité
    print("\n9. Imputation de rooms et area_m2 (similarité)...")
    df = impute_rooms_and_area(df)
    print(f"   Lignes restantes: {len(df)}")
    
    # 10. Ajout de la colonne Region
    #print("\n10. Ajout de la colonne Region...")
    #df = add_region_column(df)
    
    # 11. Imputation finale
    print("\n11. Imputation finale de rooms et area_m2...")
    df = final_imputation(df)
    
    # 12. Création de la feature luxury
    #print("\n12. Création de la feature 'is_luxury'...")
    #df = create_luxury_feature(df)
    
    # 13. Sauvegarde
    print(f"\n13. Sauvegarde des données nettoyées...")
    df.to_csv(output_filepath, index=False)
    print(f"   Dataset exporté avec succès vers: {output_filepath}")
    
    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 60)
    print(f"\nRésumé final:")
    print(f"  - Nombre de lignes: {len(df)}")
    print(f"  - Nombre de colonnes: {len(df.columns)}")
    print(f"\nValeurs manquantes par colonne:")
    missing = df.isna().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  Aucune valeur manquante!")
    
    return df


# Fonction d'utilisation principale
if __name__ == "__main__":
    # Chemins des fichiers
    input_file = r"C:\Users\MSI\housing_proj\data\raw\mubawab_1_to_275.csv"
    output_file = r"C:\Users\MSI\housing_proj\data\train\trains.csv"
    
    # Pipeline d'entraînement
    print("\n" + "="*80)
    print("TRAITEMENT DU DATASET D'ENTRAÎNEMENT")
    print("="*80)
    df_train = preprocess_pipeline(input_file, output_file)
    
    # Pour tester la compatibilité, vous pouvez créer un pipeline de test
    # en utilisant ensure_test_compatibility après avoir traité les données de test
    print("\n" + "="*80)
    print("EXEMPLE: POUR TRAITER LES DONNÉES DE TEST")
    print("="*80)
    print("""
    Pour traiter des données de test:
    
    1. Appliquer le même pipeline de nettoyage et imputation
    2. Puis appliquer ensure_test_compatibility() avant l'entraînement:
    
        # Après le preprocessing du test dataset
        df_test_processed = ensure_test_compatibility(df_test_processed, df_train)
        
    Cette fonction va:
    - Assurer que les colonnes sont identiques
    - Supprimer les types de propriété non vus en entraînement
    """)
    
    print("\nAperçu des données d'entraînement nettoyées:")
    print(df_train.head())
    print("\nInformations sur le dataset:")
    print(df_train.info())