import yaml

if __name__ == '__main__':

    entries = [
        {
            'team': 'Intel Neuromorphic Computing Lab',
            'model': 'Baseline SDNN solution',
            'date': '2023-02-20',
            'SI-SNR': 12.50,
            'SI-SNRi_data': 12.50 - 7.62,
            'SI-SNRi_enc+dec': 12.50 - 7.62,
            'MOS_ovrl': 2.71,
            'MOS_sig': 3.21,
            'MOS_bak': 3.46,
            'latency_enc+dec_ms': 0.036,
            'latency_total_ms': 8.036,
            'power_proxy_Ops/s': 11.59 * 10**6,
            'PDP_proxy_Ops': 0.09 * 10**6,
            'params': 525 * 10**3,
            'size_kilobytes': 465,
            'model_path': './baseline_solution/sdnn_delays/Trained/network.pt',
        },
        {
            'team': 'Intel Neuromorphic Computing Lab',
            'model': 'Intel proprietary DNS',
            'date': '2023-02-28',
            'SI-SNR': 12.71,
            'SI-SNRi_data': 12.71 - 7.62,
            'SI-SNRi_enc+dec': 12.71 - 7.62,
            'MOS_ovrl': 3.09,
            'MOS_sig': 3.35,
            'MOS_bak': 4.08,
            'latency_enc+dec_ms': 0.036,
            'latency_total_ms': 8.036,
            'power_proxy_Ops/s': None, 
            'PDP_proxy_Ops': None,
            'params': 1901 * 10**3,
            'size_kilobytes': 3802,
            'model_path': None,
        },
        ]
      
    with open('./metricsboard_track_1_validation.yml', 'w') as outfile:
        yaml.dump(entries, outfile)
      
