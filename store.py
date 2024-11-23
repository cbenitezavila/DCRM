def reallocate_area(df, epsg = '8857'):
    """
    Efficiently calculate the overlap between the mine polygons in each buffer and distribute the overlapping area
    equally among the mines that share the overlap.
    """

    # Initialize the area correction column
    df.to_crs(epsg=epsg, inplace=True)

    df['Territorial_overlap'] = df['geometry'].area * 10**-6  # Convert to km2

    # Process each buffer and country combination for buffer > 0
    unique_combinations = df[df['buffer']> 0] [['buffer', 'admin']].drop_duplicates()

    res = []

    for _, (b, c) in tqdm(unique_combinations.iterrows(), total=len(unique_combinations), desc='Processing Buffers, Countries'):
        try:
            # Subset data for the current buffer and country
            df_b = df[(df['buffer'] == b) & (df['admin'] == c)]

            if df_b.empty or len(df_b) < 2:
                continue  # Skip if there are no overlaps or insufficient data

            # Compute pairwise intersections
            overlaps = gpd.overlay(df_b, df_b, how='intersection')

            # Filter out self-overlaps
            overlaps = overlaps[overlaps['Mine_ID_1'] != overlaps['Mine_ID_2']]
            overlaps['Overlap_area'] = overlaps.geometry.area * 10**-6  # Convert to km2
            
            # Devide by 2 because area is shared between two overlaps
            single_share = overlaps.groupby('Mine_ID_1').apply(lambda x:  1- (x['Overlap_area'].sum() / (2*x['Territorial_overlap_1'].sum()))).reset_index()

            single_share.rename(columns={'Mine_ID_1': 'Mine_ID', 0: 'Single_share'}, inplace=True)

            assert any(single_share['Single_share'] < 0), 'Negative share detected'

            df_b = df_b.merge(single_share, on='Mine_ID', how='left')
            df_b['Territorial_overlap_corrected'] = df_b['Territorial_overlap'] * df_b['Single_share']
            
            sub = df_b[['Mine_ID', 'admin', 'buffer', 'mine_area', 'max_buffered_area', 'Territorial_overlap', 'Territorial_overlap_corrected']]
            
            res.append(sub)

            break
        
        except Exception as e:
            print(f"Error in buffer {b}, admin {c}: {e}")

    res_df = pd.DataFrame(pd.concat(res, axis = 0))

    res_df.to_csv('data\interm\corrected_overlap_area.csv', index = False)
    
    
    return df