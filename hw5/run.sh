: 'System Variation Experiments'

python3 main.py --prefix=sys_var
python3 main.py --prefix=sys_var --apply_batch_norm
python3 main.py --prefix=sys_var --apply_batch_norm --add_more_layers
python3 main.py --prefix=sys_var --apply_batch_norm --add_more_layers --larger_filter_size
python3 main.py --prefix=sys_var --apply_batch_norm --add_more_layers --larger_filter_size --add_dropout


: 'Learning Rate Experiments'

python3 main.py --prefix=lr_var --learning_rate=1e-4
python3 main.py --prefix=lr_var --learning_rate=1e-3
python3 main.py --prefix=lr_var --learning_rate=1e-2


: 'Batch Size Experiments'

python3 main.py --prefix=bs_var --batch_size=8
python3 main.py --prefix=bs_var --batch_size=64
python3 main.py --prefix=bs_var --batch_size=512
