from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1W7IGkKoiscnk_m9jpurBtsm_Cijqf28o',
                                    dest_path='./data/GTSRB.h5',
                                    unzip=True)