# Graph Neural Networks for Material Property Prediction

This project implements a Graph Neural Network (GNN) model for predicting material properties from graph-structured data.

## Step 1: Generating data ready for training GNN
### Input Directory Structure
```{bash}
python generate_train_data.py
```
```
Input_idr/
├── texture_1/
│   ├── input/                # Directory containing input features/data of texture 1
│   └── target/               # Directory containing target/label values to predict of texture 1
├── texture_2/                # Similar structure as texture_1
│   ├── input/                # Directory containing input features/data of texture 2
│   └── target/               # Directory containing target/label values to predict of texture 2
├── texture_3/                # Similar structure as texture_1
└── ...                       # Additional texture directories follow same pattern
```
### Output Directory Structure
```
Output_dir/
├── (target_scaler.pkl)           # Optional file for target value scaling
├── (input_scaler.pkl)            # Optional file for input feature scaling
├── texture_1.pkl                # Pickle file containing processed data for texture 1
├── texture_2.pkl                # Pickle file containing processed data for texture 2 
├── texture_3.pkl                # Pickle file containing processed data for texture 3
└── ...                          # Additional texture pickle files follow same pattern
```
## Step 2: Training GNN model use generated data
```{bash}
python run.py --target y --textures texture1 texture2
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--seed` | Random seed for reproducibility | 42 |
| `--target` | Property to predict | 'y' |
| `--meta-config` | Path to meta config file | './config/meta_config.json' |
| `--input-dir` | Input directory containing graph data | './graph_data/F_tensor' |
| `--output-dir` | Output directory for results | './output/F_tensor' |
| `--config-dir` | Directory containing model configs | './config/F_tensor' |
| `--verbose` | Enable verbose output | False |
| `--test-case` | Run test case | False |
| `--textures` | List of textures to process | [] |
| `--label-mask` | Label mask to process | [] |

### Target Scaling Options

Three scaling methods are available:

1. **Standard Scaling** (default)
   ```bash
   python run.py --target_scaling_method standard
   ```
   Requires `target_scaler.pickle` in input directory

2. **File Weight Scaling**
   ```bash
   python run.py --target_scaling_method file_weight --target_file_weight_filepath path/to/weights.txt
   ```

3. **Fixed Scaling**
   ```bash
   python run.py --target_scaling_method fixed --target_weight 1.0
   ```

### Grain level vs Graph level Prediction

- For graph predictions (The prediction is made on the graph level):
  ```bash
  python run.py --is_graph_level true
  ```
- For grain predictions (one prediction is made on the grain level):
  ```bash
  python run.py --is_graph_level false
  ```

## Output

The model generates:
1. Trained model checkpoints
2. Validation results log
3. Performance plots comparing predicted vs true values

## Example Usage
Only use F tensor ($3\times3$) as the input to predict sigma tensor (scalar, the top left element in $3\times3$ tensor)

### Test case for grain-level prediction and different cij (need to be updated)
```
python .\generate_train_data.py --input_dir="~/Downloads/training_raw_data/test" --input_dir_prefix="equi" --input_file_postfix .C --standard_weight_dim -1  --input_graph_postfix=".C" --target_file_postfix="_out.txt" --textures comp psc comp_rot_z-45 psc_rot_z-45 --output_filename="train_data_Cij_60.pickle" --output_dir="./graph_data/test_different_Cij" --input_file_weight_filepath="~/Downloads/100724_720/100724_720/Cij_60.txt"

python .\run.py --seed=42 --target="y" --meta-config="./config/meta_config_test.json" --input-dir="./graph_data/F_tensor" --output-dir="./output/test_case" --config-dir="./config/test_case" --verbose=True --test-case False --test-config-dir="./config/test_case" --textures 100_unrotate --label-mask 0 --target_scaling_method="standard"

python .\run.py --seed=42 --target="y" --meta-config="./config/meta_config_test.json" --input-dir="./graph_data/different_Cij/train" --output-dir="./output/test_case_cij" --config-dir="./config/test_case_cij" --verbose=True --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --target_scaling_method="standard" --use_global_prediction=true

python .\run.py --seed=42 --target="y" --meta-config="./config/meta_config_test.json" --input-dir="./graph_data/different_Cij_multi_output" --output-dir="./output/different_Cij_multi_output_test" --config-dir="./config/different_Cij_multi_output_test" --verbose=True --textures comp uni --target_scaling_method="file_weight" --target_file_weight_filepath="~/Downloads/100724_720/100724_720/Cij_60.txt" --use_global_prediction=true
```
### Different Cij for multiple output (updated)
```
python .\generate_train_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --output_dir="./graph_data/different_Cij_multi_output_21" --input_scaling_method="hybrid" --standard_weight_dim -1 --input_file_weight_filepath="~/Downloads/100724_720/100724_720/Cij_60.txt" --target_scaling_method="file_weight" --target_file_weight_filepath="~/Downloads/100724_720/100724_720/Cij_60.txt"

## use 21 outputs to train model
python .\run.py --seed=42 --target="y" --meta_config="./meta_config/meta_config_different_cij_m_o.json" --data_dir="./graph_data/different_Cij_multi_output_21" --experiment_name="different_Cij_multi_output_21_all_model" --verbose=True --textures comp uni  --target_scaling_method="file_weight" --target_file_weight_filepath="~/Downloads/100724_720/100724_720/Cij_60.txt" --is_graph_level=true

shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90

## use 21 outputs to train model, with y using standard, x using hybird
python .\generate_train_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --output_dir="./graph_data/different_Cij_x_hybird_y_standard" --input_scaling_method="hybrid" --standard_weight_dim -1 --input_file_weight_filepath="~/Downloads/100724_720/100724_720/Cij_60.txt" --target_scaling_method="standard"

python .\run.py --seed=42 --target="y" --meta_config="./meta_config/meta_config_different_cij_m_o.json" --data_dir="./graph_data/different_Cij_x_hybird_y_standard/" --experiment_name="different_Cij_x_hybird_y_standard" --verbose=True --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --target_scaling_method="standard" --is_graph_level=true

## use 21 outputs to train model, with x and y using standard
python .\generate_train_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --experiment_name="different_Cij_x_standard_y_standard" --input_scaling_method="standard" --target_scaling_method="standard"

python .\run.py --seed=42 --target="y" --meta_config="./meta_config/meta_config_different_cij_m_o.json" --data_dir="./graph_data/different_Cij_x_standard_y_standard/" --experiment_name="different_Cij_x_standard_y_standard" --verbose=True --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --target_scaling_method="standard" --is_graph_level=true

## test case 1: input uses standard, target uses standard
python .\generate_train_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni --experiment_name="test_case_1" --input_scaling_method="standard" --target_scaling_method="standard"

## test case 2: input uses file_weight, target uses standard
python .\generate_train_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni --experiment_name="test_case_2" --input_scaling_method="file_weight" --file_weight_filepath="~/Downloads/100724_720/100724_720/Cij_60.txt" --target_scaling_method="standard"

## test case 3: input uses constant, target uses standard
python .\generate_train_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni --experiment_name="test_case_3" --input_scaling_method="constant" --constant_weight 200 --standard_weight_dim -1 --target_scaling_method="standard"

## test case 4: input uses hybrid, target uses standard
python .\generate_train_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni --experiment_name="test_case_4" --input_scaling_method="hybrid" --file_weight_filepath="~/Downloads/100724_720/100724_720/Cij_60.txt" --standard_weight_dim -1 --target_scaling_method="standard"

## test case 5: input uses hybrid, target uses file_weight
python .\generate_train_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni --experiment_name="test_case_5" --input_scaling_method="hybrid" --file_weight_filepath="~/Downloads/100724_720/100724_720/Cij_60.txt" --standard_weight_dim -1 --target_scaling_method="file_weight"

## test case 6: input uses hybrid, target uses constant
python .\generate_train_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni --experiment_name="test_case_6" --input_scaling_method="hybrid" --file_weight_filepath="~/Downloads/100724_720/100724_720/Cij_60.txt" --standard_weight_dim -1 --target_scaling_method="constant" --constant_weight 200

python .\run.py --seed=42 --target="y" --meta_config="./meta_config/meta_config_test.json" --data_dir="./graph_data/different_Cij_x_standard_y_standard/" --experiment_name="test_case_6" --verbose=True --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --target_scaling_method="standard" --is_graph_level=true


## test case 7: train test split first
python .\generate_train_test_valid_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --experiment_name="test_case_7"

python .\run_train_test_model.py --meta_config="./meta_config/meta_config_common.json" --experiment_name="test_case_7" --input_scaling_method='standard' --target_scaling_method='standard' --is_graph_level=true --verbose=True

## test case 8: 5-fold validation
python .\generate_train_test_valid_data.py --input_dir="~/Downloads/720_in_out/720_in_out" --input_dir_prefix="equi" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_C_unique.txt" --textures comp uni --experiment_name="test_case_8" --n_folds=5 --test_size=0.1

## test case 9: using 100 unrotated dataset for grain-level prediction
python .\generate_train_test_valid_data.py --input_dir="~/Downloads/100_unrotate_all" --input_file_postfix .C --input_graph_postfix=".C" --target_file_postfix="_sigma_out.npy" --textures 100_unrotate --experiment_name="test_case_9"

python .\run_train_test_model.py --meta_config="./meta_config/meta_config_test.json" --experiment_name="test_case_9" --input_scaling_method='standard' --target_scaling_method='standard' --is_graph_level=False --verbose=True
```


<!-- ### F-tensor as input variables
Generating training dataset by using F tensor:
```{bash}
python .\generate_train_data.py --input_dir="~/Downloads/100_unrotate_all" --input_file_postfix="_out.npy" --target_file_postfix="_sigma_out.npy" --input_graph_postfix=".S" --textures 100_unrotate --output_dir="./graph_data/F_tensor" --input_scaling_method="standard" --target_scaling_method="standard"

python .\run.py --seed=42 --target="y" --meta-config="./config/meta_config_GAT.json" --input-dir="./graph_data/F_tensor" --output-dir="./output/F_tensor" --config-dir="./config/F_tensor" --verbose=True --textures 100_unrotate --label-mask 0 --target_scaling_method="standard"
```

F-tensor GAT attention visualization
```
python .\run.py --seed=42 --target="y" --meta-config="./config/meta_config_GAT.json" --input-dir="./graph_data/F_tensor" --output-dir="./output/GAT_F_tensor" --config-dir="./config/GAT_F_tensor" --verbose=True --test-case False --test-config-dir="./config/test_case" --textures 100_unrotate --label-mask 0 --target_scaling_method="standard"
```
### .C and .S as input variables (need update)
Generating training dataset by using features in .C and .S files:
```{bash}
python .\generate_train_data.py --input_file_postfix .C .S --target_file_postfix="_sigma_out.npy" --input_graph_postfix=".S" --textures 100_unrotate --output_filename="C_S.pickle" --output_dir="./graph_data/C_S"
```
### .C as input variables (need update)
Generating training dataset by using features in .C files:
```{bash}
python .\generate_train_data.py --input_file_postfix .C --target_file_postfix="_sigma_out.npy" --input_graph_postfix=".C" --textures 100_unrotate --output_filename="C.pickle" --output_dir="./graph_data/C"
```
### .C and .S as input variables (need update)
Generating training dataset by using features in .S files:
```{bash}
python .\generate_train_data.py --input_file_postfix .S --target_file_postfix="_sigma_out.npy" --input_graph_postfix=".S" --textures 100_unrotate --output_filename="S.pickle" --output_dir="./graph_data/S"
``` -->
<!-- ### Different cij (need update)
```
python .\generate_train_data.py --input_dir="~/Downloads/training_raw_data/720_C" --input_dir_prefix="equi" --input_file_postfix .C --input_scaling_method="hybrid" --standard_weight_dim -1 --input_file_weight_filepath="~/Downloads/training_raw_data/720_C/Cij_60.txt" --input_graph_postfix=".C" --target_file_postfix="_out.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --output_dir="./graph_data/different_Cij/train" --target_file_weight_filepath="~/Downloads/training_raw_data/720_C/Cij_60.txt"

python .\run.py --seed=42 --target="y" --meta-config="./config/meta_config_common.json" --input-dir="./graph_data/different_Cij/train" --output-dir="./output/different_Cij" --config-dir="./config/different_Cij" --verbose=True --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --target_scaling_method="standard" --use_global_prediction=true
``` -->

<!-- ### Generating test data for different metals (need update)
Cu (updated)
```
python .\generate_train_data.py --input_dir="~/Downloads/testing_raw_data/upload_Cu" --input_dir_prefix="equi" --input_file_postfix .C --standard_weight_dim -1  --input_graph_postfix=".C" --target_file_postfix="_out.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --output_dir="./graph_data/test_different_Cij_cu" --scalar_weight=168.4 --input_scaling_method="hybrid" --target_scaling_method="file_weight"

python .\run.py --seed=42 --target="y" --meta-config="./config/meta_config_GAT.json" --input-dir="./graph_data/test_different_Cij_cu" --output-dir="./output/different_Cij_cu" --config-dir="./config/different_Cij_cu" --verbose=True --test-case False --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --target_scaling_method="fixed" --target_weight=168.4 --use_global_prediction=True
```

Fe
```
python .\generate_train_data.py --input_dir="~/Downloads/testing_raw_data/upload_Fe" --input_dir_prefix="equi" --input_file_postfix .C --standard_weight_dim -1  --input_graph_postfix=".C" --target_file_postfix="_out.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --output_filename="test_fe_Cij.pickle" --output_dir="./graph_data/test_different_Cij_fe" --scalar_weight=232.2
```

Ni (updated)
```
python .\generate_train_data.py --input_dir="~/Downloads/testing_raw_data/upload_Ni" --input_dir_prefix="equi" --input_file_postfix .C --standard_weight_dim -1  --input_graph_postfix=".C" --target_file_postfix="_out.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --output_dir="./graph_data/different_Cij_Ni" --scalar_weight=267.1 --input_scaling_method="hybrid" --target_scaling_method="file_weight"

python .\run.py --seed=42 --target="y" --meta-config="./config/meta_config_common.json" --input-dir="./graph_data/different_Cij_Ni" --output-dir="./output/different_Cij_Ni" --config-dir="./config/different_Cij_Ni" --verbose=True --test-case False --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --target_scaling_method="fixed" --target_weight=168.4 --use_global_prediction=True
```

Nb
```
python .\generate_train_data.py --input_dir="~/Downloads/testing_raw_data/upload_Nb" --input_dir_prefix="equi" --input_file_postfix .C --standard_weight_dim -1  --input_graph_postfix=".C" --target_file_postfix="_out.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --output_filename="test_nb_Cij.pickle" --output_dir="./graph_data/test_different_Cij_nb" --scalar_weight=246
```

Mg
```
python .\generate_train_data.py --input_dir="~/Downloads/testing_raw_data/upload_Mg" --input_dir_prefix="equi" --input_file_postfix .C --standard_weight_dim -1  --input_graph_postfix=".C" --target_file_postfix="_out.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --output_filename="test_mg_Cij.pickle" --output_dir="./graph_data/test_different_Cij_mg" --scalar_weight=59.4
```

Ti
```
python .\generate_train_data.py --input_dir="~/Downloads/testing_raw_data/upload_Ti" --input_dir_prefix="equi" --input_file_postfix .C --standard_weight_dim -1  --input_graph_postfix=".C" --target_file_postfix="_out.txt" --textures comp uni shear psc comp_rot_z-45 uni_rot_z-45 shear_rot_z-45 psc_rot_z-45 comp_rot_z-90 uni_rot_z-90 shear_rot_z-90 psc_rot_z-90 --output_filename="test_ti_Cij.pickle" --output_dir="./graph_data/test_different_Cij_ti" --scalar_weight=162.4
``` -->
