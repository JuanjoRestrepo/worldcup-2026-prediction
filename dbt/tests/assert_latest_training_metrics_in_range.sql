select *
from {{ ref('gold_latest_training_run') }}
where accuracy < 0
   or accuracy > 1
   or macro_f1 < 0
   or macro_f1 > 1
   or weighted_f1 < 0
   or weighted_f1 > 1
   or log_loss < 0
