import time, os
from django.conf import settings
from utility.logging import init_logging, logging

right_to_draw_logs = init_logging(log_name='right_to_draw_logs',
                                  log_level=logging.DEBUG,
                                  rotation_criteria='time',
                                  rotate_interval=1,
                                  rotate_when='d',
                                  backup_count=30,
                                  log_directory=os.path.join(settings.BASE_DIR, "logs"))