========================================================================
Domain filtering A/B report -- 80 questions
========================================================================
errors:  with=0  without=0

Domain prediction accuracy (against the eval set's expected_domain):
  scoped & correct:     56
  scoped & WRONG:       20
  predicted full graph: 4  (no scoping attempted)
  accuracy when scoped: 0.7368

With-filter vs without-filter deltas (with minus without):
  avg confidence delta:   -0.0008
  avg elapsed_s delta:    -4.237
  avg graph-chunk delta:  -9.6667
  avg reference overlap:  0.4372  (Jaccard, 1.0 = identical sources)
  % answers changed:      98.8%

LLM-as-judge scores (1-5, pointwise per answer, averaged):
  dimension         with   without     delta
  completeness       4.8     4.838    -0.038
  relevance        4.925      4.95    -0.025
  fluency          4.987      4.95     0.037
  accuracy         4.775     4.775       0.0
  overall          4.872     4.878    -0.006

Questions where the predicted domain MISSED the expected one (20):
  [ADVANCED_SYSTEMS-02] expected='ADVANCED SYSTEMS' predicted=['CONTROL SYSTEMS', 'INFORMATION TECHNOLOGY', 'PROJECT EXECUTION SYSTEMS']
  [COSTS_CONTROL-02] expected='COSTS CONTROL' predicted=['FINANCE / TAX']
  [CYBERSECURITY-01] expected='CYBERSECURITY' predicted=['CONTROL SYSTEMS']
  [CYBERSECURITY-02] expected='CYBERSECURITY' predicted=['INFORMATION TECHNOLOGY']
  [FIRED_EQUIPMENT-01] expected='FIRED EQUIPMENT' predicted=['PROCESS & HSED (SAFETY IN DESIGN)', 'PACKAGE']
  [FIRED_EQUIPMENT-02] expected='FIRED EQUIPMENT' predicted=['PROCESS & HSED (SAFETY IN DESIGN)']
  [INFO_TECH-02] expected='INFORMATION TECHNOLOGY' predicted=['INFORMATION MANAGEMENT', 'PROJECT EXECUTION SYSTEMS']
  [IP_NDA-01] expected='INTELLECTUAL PROPERTY / NDA' predicted=['LEGAL / CONTRACT']
  [IP_NDA-02] expected='INTELLECTUAL PROPERTY / NDA' predicted=['LEGAL / CONTRACT']
  [LABORATORY-01] expected='LABORATORY' predicted=['QUALITY MANAGEMENT']
  [NAVAL_ARCHITECTURE-01] expected='NAVAL ARCHITECTURE' predicted=['WEIGHT CONTROL AND MANAGEMENT']
  [NAVAL_ARCHITECTURE-02] expected='NAVAL ARCHITECTURE' predicted=['PRESSURE VESSELS']
  [OFFSHORE_BUILDING-01] expected='OFFSHORE BUILDING/ ARCHITECTURE' predicted=['ONSHORE BUILDING/ ARCHITECTURE', 'CIVIL / STRUCTURAL', 'HVAC']
  [OFFSHORE_BUILDING-02] expected='OFFSHORE BUILDING/ ARCHITECTURE' predicted=['ONSHORE BUILDING/ ARCHITECTURE']
  [OFFSHORE_MOBILE-01] expected='OFFSHORE MOBILE SYSTEM' predicted=['PROCESS & HSED (SAFETY IN DESIGN)', 'CIVIL / STRUCTURAL', 'QUALITY MANAGEMENT']
  [OFFSHORE_MOBILE-02] expected='OFFSHORE MOBILE SYSTEM' predicted=['PROCESS & HSED (SAFETY IN DESIGN)', 'CIVIL / STRUCTURAL', 'WEIGHT CONTROL AND MANAGEMENT']
  [OFFSHORE_STRUCTURE-01] expected='OFFSHORE STRUCTURE' predicted=['CIVIL / STRUCTURAL']
  [OFFSHORE_STRUCTURE-02] expected='OFFSHORE STRUCTURE' predicted=['PIPELINES']
  [TRANSPORT_INSTALLATION-01] expected='TRANSPORT AND INSTALLATION' predicted=['SUBCONTRACTING', 'CIVIL / STRUCTURAL']
  [TRANSPORT_INSTALLATION-02] expected='TRANSPORT AND INSTALLATION' predicted=['SUBCONTRACTING', 'CIVIL / STRUCTURAL']




  ========================================================================
Domain filtering A/B report -- 78 questions
========================================================================
errors:  with=0  without=2

Domain prediction accuracy (against the eval set's expected_domain):
  scoped & correct:     56
  scoped & WRONG:       22
  predicted full graph: 0  (no scoping attempted)
  accuracy when scoped: 0.7179

With-filter vs without-filter deltas (with minus without):
  avg confidence delta:   -0.1107
  avg elapsed_s delta:    -2.8209
  avg graph-chunk delta:  -10.1974
  avg reference overlap:  0.3035  (Jaccard, 1.0 = identical sources)
  % answers changed:      100.0%

LLM-as-judge scores (1-5, pointwise per answer, averaged):
  dimension         with   without     
  completeness     4.211     4.776    
  relevance        4.513     4.974    
  fluency          4.737     4.974    
  accuracy         4.276     4.711   
  overall          4.434     4.859    

Questions where the predicted domain MISSED the expected one (22):
  [ADVANCED_SYSTEMS-01] expected='ADVANCED SYSTEMS' predicted=['INFORMATION TECHNOLOGY', 'CONTROL SYSTEMS']
  [ADVANCED_SYSTEMS-02] expected='ADVANCED SYSTEMS' predicted=['CONTROL SYSTEMS', 'PROJECT EXECUTION SYSTEMS']
  [CYBERSECURITY-01] expected='CYBERSECURITY' predicted=['CONTROL SYSTEMS', 'INFORMATION TECHNOLOGY']
  [CYBERSECURITY-02] expected='CYBERSECURITY' predicted=['INFORMATION TECHNOLOGY']
  [FIRED_EQUIPMENT-01] expected='FIRED EQUIPMENT' predicted=['PROCESS & HSED (SAFETY IN DESIGN)', 'PACKAGE']
  [FIRED_EQUIPMENT-02] expected='FIRED EQUIPMENT' predicted=['PROCESS & HSED (SAFETY IN DESIGN)']
  [IP_NDA-01] expected='INTELLECTUAL PROPERTY / NDA' predicted=['LEGAL / CONTRACT']
  [IP_NDA-02] expected='INTELLECTUAL PROPERTY / NDA' predicted=['LEGAL / CONTRACT']
  [LABORATORY-01] expected='LABORATORY' predicted=['QUALITY MANAGEMENT']
  [LABORATORY-02] expected='LABORATORY' predicted=['PACKAGE']
  [NAVAL_ARCHITECTURE-01] expected='NAVAL ARCHITECTURE' predicted=['WEIGHT CONTROL AND MANAGEMENT']
  [NAVAL_ARCHITECTURE-02] expected='NAVAL ARCHITECTURE' predicted=['CIVIL / STRUCTURAL']
  [OFFSHORE_BUILDING-01] expected='OFFSHORE BUILDING/ ARCHITECTURE' predicted=['ONSHORE BUILDING/ ARCHITECTURE', 'HVAC']
  [OFFSHORE_BUILDING-02] expected='OFFSHORE BUILDING/ ARCHITECTURE' predicted=['ONSHORE BUILDING/ ARCHITECTURE']
  [OFFSHORE_MOBILE-01] expected='OFFSHORE MOBILE SYSTEM' predicted=['PROCESS & HSED (SAFETY IN DESIGN)']
  [OFFSHORE_MOBILE-02] expected='OFFSHORE MOBILE SYSTEM' predicted=['PROCESS & HSED (SAFETY IN DESIGN)']
  [OFFSHORE_STRUCTURE-01] expected='OFFSHORE STRUCTURE' predicted=['CIVIL / STRUCTURAL']
  [OFFSHORE_STRUCTURE-02] expected='OFFSHORE STRUCTURE' predicted=['CIVIL / STRUCTURAL']
  [PRESSURE_VESSELS-01] expected='PRESSURE VESSELS' predicted=['PACKAGE']
  [PRESSURE_VESSELS-02] expected='PRESSURE VESSELS' predicted=['ROTATING EQUIPMENT', 'QUALITY MANAGEMENT']
  [TRANSPORT_INSTALLATION-01] expected='TRANSPORT AND INSTALLATION' predicted=['WEIGHT CONTROL AND MANAGEMENT', 'CONSTRUCTION AND PCC&SU']
  [TRANSPORT_INSTALLATION-02] expected='TRANSPORT AND INSTALLATION' predicted=['CONSTRUCTION AND PCC&SU', 'SUBCONTRACTING']






  Domain prediction accuracy (against the eval set's expected_domain):
  scoped & correct:     54
  scoped & WRONG:       24
  predicted full graph: 0  (no scoping attempted)
  accuracy when scoped: 0.6923

With-filter vs without-filter deltas (with minus without):
  avg confidence delta:   -0.0094
  avg elapsed_s delta:    -2.5587
  avg graph-chunk delta:  -6.9481
  avg reference overlap:  0.4649  (Jaccard, 1.0 = identical sources)
  % answers changed:      100.0%

LLM-as-judge scores (1-5, pointwise per answer, averaged):
  dimension         with   without     delta
  completeness     4.688     4.818     -0.13
  relevance        4.844     4.922    -0.078
  fluency          4.948     4.961    -0.013
  accuracy         4.727     4.766    -0.039
  overall          4.802     4.867    -0.065

Questions where the predicted domain MISSED the expected one (24):
  [ADVANCED_SYSTEMS-01] expected='ADVANCED SYSTEMS' predicted=['INFORMATION TECHNOLOGY', 'CONTROL SYSTEMS']
  [ADVANCED_SYSTEMS-02] expected='ADVANCED SYSTEMS' predicted=['CONTROL SYSTEMS', 'PROJECT EXECUTION SYSTEMS']
  [CIVIL_STRUCTURAL-01] expected='CIVIL / STRUCTURAL' predicted=[]
  [CYBERSECURITY-01] expected='CYBERSECURITY' predicted=['CONTROL SYSTEMS', 'INFORMATION TECHNOLOGY']
  [CYBERSECURITY-02] expected='CYBERSECURITY' predicted=['INFORMATION TECHNOLOGY']
  [FIRED_EQUIPMENT-01] expected='FIRED EQUIPMENT' predicted=['PROCESS & HSED (SAFETY IN DESIGN)', 'PACKAGE']
  [FIRED_EQUIPMENT-02] expected='FIRED EQUIPMENT' predicted=['PROCESS & HSED (SAFETY IN DESIGN)']
  [INFO_TECH-02] expected='INFORMATION TECHNOLOGY' predicted=['PROJECT EXECUTION SYSTEMS', 'INFORMATION MANAGEMENT']
  [IP_NDA-01] expected='INTELLECTUAL PROPERTY / NDA' predicted=['LEGAL / CONTRACT']
  [IP_NDA-02] expected='INTELLECTUAL PROPERTY / NDA' predicted=['LEGAL / CONTRACT']
  [LABORATORY-01] expected='LABORATORY' predicted=['QUALITY MANAGEMENT']
  [LABORATORY-02] expected='LABORATORY' predicted=['PACKAGE']
  [NAVAL_ARCHITECTURE-01] expected='NAVAL ARCHITECTURE' predicted=['WEIGHT CONTROL AND MANAGEMENT']
  [NAVAL_ARCHITECTURE-02] expected='NAVAL ARCHITECTURE' predicted=['CIVIL / STRUCTURAL']
  [OFFSHORE_BUILDING-01] expected='OFFSHORE BUILDING/ ARCHITECTURE' predicted=['ONSHORE BUILDING/ ARCHITECTURE', 'HVAC']
  [OFFSHORE_BUILDING-02] expected='OFFSHORE BUILDING/ ARCHITECTURE' predicted=['ONSHORE BUILDING/ ARCHITECTURE']
  [OFFSHORE_MOBILE-01] expected='OFFSHORE MOBILE SYSTEM' predicted=['PROCESS & HSED (SAFETY IN DESIGN)']
  [OFFSHORE_MOBILE-02] expected='OFFSHORE MOBILE SYSTEM' predicted=['PROCESS & HSED (SAFETY IN DESIGN)']
  [OFFSHORE_STRUCTURE-01] expected='OFFSHORE STRUCTURE' predicted=['CIVIL / STRUCTURAL']
  [OFFSHORE_STRUCTURE-02] expected='OFFSHORE STRUCTURE' predicted=['ROTATING EQUIPMENT']
  [PRESSURE_VESSELS-01] expected='PRESSURE VESSELS' predicted=['PACKAGE']
  [PRESSURE_VESSELS-02] expected='PRESSURE VESSELS' predicted=['ROTATING EQUIPMENT', 'QUALITY MANAGEMENT']
  [TRANSPORT_INSTALLATION-01] expected='TRANSPORT AND INSTALLATION' predicted=['WEIGHT CONTROL AND MANAGEMENT', 'CONSTRUCTION AND PCC&SU']
  [TRANSPORT_INSTALLATION-02] expected='TRANSPORT AND INSTALLATION' predicted=['CONSTRUCTION AND PCC&SU']