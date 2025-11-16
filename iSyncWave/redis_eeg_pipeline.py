#!/usr/bin/env python3
"""
Redis EEG Data Processing Pipeline
Nature Communications 2025 논문 방법을 Redis 데이터에 적용

타임스탬프 마커 없이 연속적인 EEG 데이터를 처리합니다.
"""

import redis
import numpy as np
import mne
from scipy import signal
from scipy.signal import welch
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import pickle
from datetime import datetime, timedelta


class RedisEEGPipeline:
    """Redis에서 EEG 데이터를 읽어 논문 방법으로 전처리하는 파이프라인"""

    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """
        Parameters:
        -----------
        redis_host : str
            Redis 서버 호스트
        redis_port : int
            Redis 서버 포트
        redis_db : int
            Redis 데이터베이스 번호
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.data = None
        self.processed_data = None
        self.segments = None
        self.features = None

    def connect(self):
        """Redis 연결 확인"""
        try:
            self.redis_client.ping()
            print("✓ Redis 연결 성공")
            return True
        except Exception as e:
            print(f"❌ Redis 연결 실패: {e}")
            return False

    def _load_by_time_range(self, start_time, end_time):
        """
        시간 범위로 데이터 로드 (내부 헬퍼 함수)

        Parameters:
        -----------
        start_time : datetime or float or None
            시작 시간
        end_time : datetime or float or None
            종료 시간

        Returns:
        --------
        list : Redis stream 데이터
        """
        # 전체 데이터 범위 확인
        time_range = self.get_available_time_range()
        if time_range is None:
            return []

        # LSL 타임스탬프로 변환
        if start_time is None:
            start_lsl = time_range['start_timestamp']
        elif isinstance(start_time, (int, float)):
            start_lsl = float(start_time)
        elif isinstance(start_time, datetime):
            # datetime을 LSL 타임스탬프로 변환 (상대적 계산)
            dt_diff = (start_time - time_range['start_time']).total_seconds()
            start_lsl = time_range['start_timestamp'] + dt_diff
        else:
            print(f"⚠️  지원하지 않는 start_time 타입: {type(start_time)}")
            start_lsl = time_range['start_timestamp']

        if end_time is None:
            end_lsl = time_range['end_timestamp']
        elif isinstance(end_time, (int, float)):
            end_lsl = float(end_time)
        elif isinstance(end_time, datetime):
            # datetime을 LSL 타임스탬프로 변환 (상대적 계산)
            dt_diff = (end_time - time_range['start_time']).total_seconds()
            end_lsl = time_range['start_timestamp'] + dt_diff
        else:
            print(f"⚠️  지원하지 않는 end_time 타입: {type(end_time)}")
            end_lsl = time_range['end_timestamp']

        print(f"\n시간 범위 필터:")
        print(f"  요청 LSL 범위: {start_lsl:.3f} ~ {end_lsl:.3f} ({end_lsl - start_lsl:.2f}초)")

        # 전체 데이터를 가져와서 LSL 타임스탬프로 필터링
        # (Redis XRANGE는 Stream ID 기반이므로, LSL 타임스탬프로 필터링하려면 전체를 가져와야 함)
        all_data = self.redis_client.xrange('isyncwave:eeg:stream', '-', '+')

        filtered_data = []
        for msg_id, data in all_data:
            lsl_ts = float(data.get('lsl_timestamp', 0))
            if start_lsl <= lsl_ts <= end_lsl:
                filtered_data.append((msg_id, data))

        print(f"  필터링된 샘플: {len(filtered_data)} / {len(all_data)}")

        return filtered_data

    def get_available_time_range(self):
        """
        Redis에 저장된 데이터의 시간 범위 확인

        Returns:
        --------
        dict : 시간 범위 정보
            - 'start_time': 시작 시간 (datetime)
            - 'end_time': 종료 시간 (datetime)
            - 'start_timestamp': 시작 LSL 타임스탬프
            - 'end_timestamp': 종료 LSL 타임스탬프
            - 'duration': 총 시간 (초)
            - 'total_samples': 총 샘플 수
        """
        print("\n" + "="*90)
        print("사용 가능한 데이터 시간 범위 확인")
        print("="*90)

        # 첫 번째와 마지막 데이터 조회
        first_data = self.redis_client.xrange('isyncwave:eeg:stream', '-', '+', count=1)
        last_data = self.redis_client.xrevrange('isyncwave:eeg:stream', '+', '-', count=1)

        if not first_data or not last_data:
            print("❌ 스트림 데이터가 없습니다!")
            return None

        # 타임스탬프 추출
        first_id, first_values = first_data[0]
        last_id, last_values = last_data[0]

        start_lsl_ts = float(first_values.get('lsl_timestamp', 0))
        end_lsl_ts = float(last_values.get('lsl_timestamp', 0))

        # Stream ID에서 Redis 타임스탬프 추출 (밀리초)
        first_redis_ts = int(first_id.split('-')[0])
        last_redis_ts = int(last_id.split('-')[0])

        # datetime 변환
        start_dt = datetime.fromtimestamp(first_redis_ts / 1000)
        end_dt = datetime.fromtimestamp(last_redis_ts / 1000)

        # 총 샘플 수 확인
        total_samples = self.redis_client.xlen('isyncwave:eeg:stream')

        duration = end_lsl_ts - start_lsl_ts

        print(f"\n데이터 범위:")
        print(f"  시작 시간: {start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print(f"  종료 시간: {end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print(f"  총 시간: {duration:.2f}초 ({duration/60:.2f}분)")
        print(f"  총 샘플 수: {total_samples:,}")
        print(f"  LSL 타임스탬프 범위: {start_lsl_ts:.3f} ~ {end_lsl_ts:.3f}")

        return {
            'start_time': start_dt,
            'end_time': end_dt,
            'start_timestamp': start_lsl_ts,
            'end_timestamp': end_lsl_ts,
            'start_redis_id': first_id,
            'end_redis_id': last_id,
            'duration': duration,
            'total_samples': total_samples
        }

    def load_data(self, start_time=None, end_time=None, count=None):
        """
        Redis에서 EEG 데이터 로드

        Parameters:
        -----------
        start_time : datetime or float, optional
            시작 시간 (datetime 객체 또는 LSL 타임스탬프)
            None이면 처음부터 로드
        end_time : datetime or float, optional
            종료 시간 (datetime 객체 또는 LSL 타임스탬프)
            None이면 끝까지 로드
        count : int, optional
            읽을 최대 샘플 수 (start_time/end_time보다 우선순위 낮음)
            start_time/end_time이 None일 때만 사용

        Returns:
        --------
        bool : 성공 여부

        Examples:
        ---------
        # 최근 10000개 샘플 로드
        pipeline.load_data(count=10000)

        # 특정 시간 범위 로드 (LSL 타임스탬프)
        pipeline.load_data(start_time=100.0, end_time=200.0)

        # 특정 시간 범위 로드 (datetime)
        from datetime import datetime, timedelta
        now = datetime.now()
        pipeline.load_data(start_time=now-timedelta(minutes=5), end_time=now)
        """
        print("\n" + "="*90)
        print("REDIS 데이터 로드")
        print("="*90)

        # 메타데이터 읽기
        meta = self.redis_client.hgetall('isyncwave:eeg:meta')
        if not meta:
            print("❌ Redis에 메타데이터가 없습니다!")
            return False

        print(f"\n메타데이터:")
        print(f"  채널 수: {meta.get('channel_count', 'N/A')}")
        print(f"  샘플링 레이트: {meta.get('sampling_rate', 'N/A')} Hz")
        print(f"  채널 이름: {meta.get('channels', 'N/A')}")

        # 시간 범위에 따른 데이터 조회 방법 결정
        if start_time is not None or end_time is not None:
            # 시간 범위 기반 조회
            stream_data = self._load_by_time_range(start_time, end_time)
        else:
            # count 기반 조회 (기존 방식)
            if count is None:
                count = 100000
            stream_data = self.redis_client.xrevrange(
                'isyncwave:eeg:stream', '+', '-', count=count
            )

        if not stream_data:
            print("❌ 스트림 데이터가 없습니다!")
            return False

        print(f"\n스트림 데이터:")
        print(f"  로드된 샘플: {len(stream_data)}")

        # 채널 파싱
        channel_names = meta.get('channels', '').split(',')
        channel_names = [ch.strip() for ch in channel_names if ch.strip()]

        print(f"  파싱된 채널 ({len(channel_names)}): {', '.join(channel_names)}")

        # 데이터 추출
        signals = {ch: [] for ch in channel_names}
        timestamps = []

        # 시간순으로 정렬
        # xrevrange 사용 시: 역순 정렬되어 있으므로 reversed 필요
        # xrange 사용 시 (시간 범위 조회): 이미 정순 정렬
        if start_time is not None or end_time is not None:
            # 시간 범위 조회 - 이미 정순
            data_iter = stream_data
        else:
            # count 기반 조회 (xrevrange) - 역순이므로 뒤집기
            data_iter = reversed(stream_data)

        for msg_id, data in data_iter:
            timestamps.append(float(data.get('lsl_timestamp', 0)))
            for ch in channel_names:
                if ch in data:
                    signals[ch].append(float(data[ch]))

        # numpy 배열로 변환
        signal_arrays = np.array([signals[ch] for ch in channel_names])

        sampling_rate = float(meta.get('sampling_rate', 250))
        duration = len(timestamps) / sampling_rate

        print(f"\n데이터 로드 완료:")
        print(f"  Shape: {signal_arrays.shape} (channels x samples)")
        print(f"  Duration: {duration:.2f} 초")
        print(f"  Sampling rate: {sampling_rate} Hz")

        self.data = {
            'channels': channel_names,
            'signals': signal_arrays,  # μV 단위
            'sampling_rate': sampling_rate,
            'n_samples': len(timestamps),
            'timestamps': np.array(timestamps),
            'duration': duration
        }

        return True

    def preprocess(self, apply_ica=True):
        """
        Nature Communications 2025 논문 전처리 적용

        전처리 순서:
        1. Common Average Reference (CAR)
        2. Notch filter (50Hz, 60Hz)
        3. Downsample to 100 Hz
        4. Bandpass filter (4-40 Hz, 4th order Butterworth)
        5. ICA artifact removal (optional)

        Parameters:
        -----------
        apply_ica : bool
            ICA 아티팩트 제거 적용 여부

        Returns:
        --------
        bool : 성공 여부
        """
        if self.data is None:
            print("❌ 먼저 load_data()를 실행하세요!")
            return False

        print("\n" + "="*90)
        print("NATURE COMMUNICATIONS 2025 논문 전처리 파이프라인 적용")
        print("="*90)

        # MNE Raw 객체 생성
        n_channels = len(self.data['channels'])
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(
            ch_names=self.data['channels'],
            sfreq=self.data['sampling_rate'],
            ch_types=ch_types
        )

        # μV → V 변환 (MNE는 V 단위 사용)
        raw = mne.io.RawArray(self.data['signals'] / 1e6, info, verbose=False)

        # Step 1: Common Average Reference (CAR)
        print("\n[1/5] Common Average Reference (CAR) 적용...")
        raw.set_eeg_reference('average', projection=False, verbose=False)

        # Step 2: Notch filter (50Hz, 60Hz)
        print("[2/5] Notch Filter 적용 (50Hz, 60Hz)...")
        raw.notch_filter(freqs=[50, 60], verbose=False)

        # Step 3: Downsample to 100 Hz
        print("[3/5] 다운샘플링 (100 Hz)...")
        original_sfreq = raw.info['sfreq']
        raw.resample(100, verbose=False)
        print(f"   {original_sfreq} Hz → 100 Hz")

        # Step 4: Bandpass filter (4-40 Hz, 4th order Butterworth)
        print("[4/5] Bandpass Filter 적용 (4-40 Hz, 4차 Butterworth)...")
        raw.filter(
            l_freq=4.0,
            h_freq=40.0,
            method='iir',
            iir_params={'order': 4, 'ftype': 'butter'},
            verbose=False
        )

        # Step 5: ICA artifact removal
        if apply_ica:
            print("[5/5] ICA 아티팩트 제거...")
            try:
                # ICA 피팅용 복사본 (1-30 Hz 필터)
                raw_ica = raw.copy()
                raw_ica.filter(l_freq=1.0, h_freq=30.0, verbose=False)

                # ICA 적용
                ica = ICA(n_components=15, random_state=42, max_iter=500, verbose=False)
                ica.fit(raw_ica, verbose=False)

                # EOG 아티팩트 자동 검출
                eog_indices, eog_scores = ica.find_bads_eog(raw_ica, verbose=False)

                if len(eog_indices) > 0:
                    print(f"   {len(eog_indices)}개 EOG 컴포넌트 발견: {eog_indices}")
                    ica.exclude = eog_indices
                    ica.apply(raw, verbose=False)
                else:
                    print("   명확한 EOG 컴포넌트를 찾지 못했습니다")

            except Exception as e:
                print(f"   ICA 실패: {e}, ICA 단계 건너뜀")
        else:
            print("[5/5] ICA 건너뜀 (apply_ica=False)")

        # 전처리된 데이터 저장 (V → μV)
        processed_signals = raw.get_data() * 1e6

        print("\n✓ 전처리 완료!")
        print(f"  최종 샘플링 레이트: {raw.info['sfreq']} Hz")
        print(f"  최종 데이터 Shape: {processed_signals.shape}")

        self.processed_data = {
            'signals': processed_signals,
            'sampling_rate': raw.info['sfreq'],
            'channels': self.data['channels'],
            'duration': processed_signals.shape[1] / raw.info['sfreq']
        }

        return True

    def segment_data(self, window_size=1.0, step_size=0.125):
        """
        데이터를 윈도우로 세그먼트화하고 z-score 정규화

        Parameters:
        -----------
        window_size : float
            윈도우 크기 (초) - 논문에서는 1.0초 사용
        step_size : float
            스텝 크기 (초) - 논문에서는 0.125초 (125ms) 사용

        Returns:
        --------
        bool : 성공 여부
        """
        if self.processed_data is None:
            print("❌ 먼저 preprocess()를 실행하세요!")
            return False

        print("\n" + "="*90)
        print("데이터 세그먼트화 및 정규화")
        print("="*90)

        data = self.processed_data['signals']
        sfreq = self.processed_data['sampling_rate']
        n_channels, n_samples = data.shape

        # 윈도우 및 스텝 크기 (샘플 수)
        window_samples = int(window_size * sfreq)
        step_samples = int(step_size * sfreq)

        print(f"\n파라미터:")
        print(f"  윈도우 크기: {window_size}초 ({window_samples} 샘플)")
        print(f"  스텝 크기: {step_size}초 ({step_samples} 샘플)")

        # 세그먼트 생성
        segments = []
        segment_times = []

        for start_idx in range(0, n_samples - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            segment = data[:, start_idx:end_idx]

            # Z-score 정규화 (채널별)
            segment_normalized = np.zeros_like(segment)
            for ch_idx in range(n_channels):
                mean = np.mean(segment[ch_idx, :])
                std = np.std(segment[ch_idx, :])
                if std > 0:
                    segment_normalized[ch_idx, :] = (segment[ch_idx, :] - mean) / std
                else:
                    segment_normalized[ch_idx, :] = segment[ch_idx, :]

            segments.append(segment_normalized)
            segment_times.append(start_idx / sfreq)

        segments = np.array(segments)  # Shape: (n_segments, n_channels, window_samples)

        print(f"\n세그먼트화 완료:")
        print(f"  총 세그먼트 수: {len(segments)}")
        print(f"  세그먼트 Shape: {segments.shape}")

        self.segments = {
            'data': segments,
            'times': np.array(segment_times),
            'window_size': window_size,
            'step_size': step_size,
            'sampling_rate': sfreq
        }

        return True

    def extract_features(self, alpha_band=(8, 13), beta_band=(13, 30)):
        """
        각 세그먼트에서 ERD 특징 추출

        Parameters:
        -----------
        alpha_band : tuple
            알파 밴드 주파수 범위 (Hz)
        beta_band : tuple
            베타 밴드 주파수 범위 (Hz)

        Returns:
        --------
        bool : 성공 여부
        """
        if self.segments is None:
            print("❌ 먼저 segment_data()를 실행하세요!")
            return False

        print("\n" + "="*90)
        print("특징 추출 (ERD)")
        print("="*90)

        segments = self.segments['data']
        sfreq = self.segments['sampling_rate']
        n_segments, n_channels, n_samples = segments.shape

        print(f"\n밴드 정의:")
        print(f"  알파 밴드: {alpha_band[0]}-{alpha_band[1]} Hz")
        print(f"  베타 밴드: {beta_band[0]}-{beta_band[1]} Hz")

        # 특징 배열 초기화
        alpha_power = np.zeros((n_segments, n_channels))
        beta_power = np.zeros((n_segments, n_channels))

        # 각 세그먼트에 대해 PSD 계산
        for seg_idx in range(n_segments):
            for ch_idx in range(n_channels):
                # Welch 방법으로 PSD 계산
                freqs, psd = welch(
                    segments[seg_idx, ch_idx, :],
                    fs=sfreq,
                    nperseg=min(256, n_samples)
                )

                # 알파 밴드 파워
                alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
                alpha_power[seg_idx, ch_idx] = np.mean(psd[alpha_mask])

                # 베타 밴드 파워
                beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
                beta_power[seg_idx, ch_idx] = np.mean(psd[beta_mask])

        print(f"\n특징 추출 완료:")
        print(f"  알파 파워 Shape: {alpha_power.shape}")
        print(f"  베타 파워 Shape: {beta_power.shape}")

        self.features = {
            'alpha_power': alpha_power,
            'beta_power': beta_power,
            'alpha_band': alpha_band,
            'beta_band': beta_band,
            'n_segments': n_segments,
            'n_channels': n_channels,
            'segment_times': self.segments['times']
        }

        return True

    def save_processed_data(self, output_path=None):
        """
        전처리된 데이터와 특징을 저장

        Parameters:
        -----------
        output_path : str
            저장 경로 (None이면 자동 생성)

        Returns:
        --------
        str : 저장된 파일 경로
        """
        if self.processed_data is None or self.segments is None or self.features is None:
            print("❌ 먼저 전처리, 세그먼트화, 특징 추출을 완료하세요!")
            return None

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"processed_eeg_data_{timestamp}.pkl"

        # 저장할 데이터 패키징
        save_data = {
            'original_data': self.data,
            'processed_data': self.processed_data,
            'segments': self.segments,
            'features': self.features,
            'pipeline_info': {
                'preprocessing': 'Nature Communications 2025',
                'timestamp': datetime.now().isoformat()
            }
        }

        # Pickle로 저장
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"\n✓ 데이터 저장 완료: {output_path}")

        # 저장된 내용 요약
        print(f"\n저장된 내용:")
        print(f"  원본 데이터: {self.data['signals'].shape}")
        print(f"  전처리 데이터: {self.processed_data['signals'].shape}")
        print(f"  세그먼트 수: {self.segments['data'].shape[0]}")
        print(f"  특징 (알파): {self.features['alpha_power'].shape}")
        print(f"  특징 (베타): {self.features['beta_power'].shape}")

        return output_path

    def visualize_results(self, save_path='redis_pipeline_results.png'):
        """
        분석 결과 시각화

        Parameters:
        -----------
        save_path : str
            저장할 이미지 경로
        """
        if self.features is None:
            print("❌ 먼저 특징 추출을 완료하세요!")
            return

        print(f"\n시각화 생성 중...")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. 전처리된 신호 (첫 번째 채널)
        ax1 = fig.add_subplot(gs[0, :])
        times = np.arange(self.processed_data['signals'].shape[1]) / self.processed_data['sampling_rate']
        ax1.plot(times, self.processed_data['signals'][0, :], linewidth=0.5)
        ax1.set_xlabel('시간 (초)', fontweight='bold')
        ax1.set_ylabel('진폭 (μV)', fontweight='bold')
        ax1.set_title(f'전처리된 EEG 신호 - {self.processed_data["channels"][0]}',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. 알파 파워 시계열 (모든 채널 평균)
        ax2 = fig.add_subplot(gs[1, 0])
        alpha_mean = np.mean(self.features['alpha_power'], axis=1)
        ax2.plot(self.features['segment_times'], alpha_mean, linewidth=1.5, color='steelblue')
        ax2.set_xlabel('시간 (초)', fontweight='bold')
        ax2.set_ylabel('알파 파워', fontweight='bold')
        ax2.set_title('알파 밴드 파워 (8-13 Hz) - 채널 평균', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. 베타 파워 시계열 (모든 채널 평균)
        ax3 = fig.add_subplot(gs[1, 1])
        beta_mean = np.mean(self.features['beta_power'], axis=1)
        ax3.plot(self.features['segment_times'], beta_mean, linewidth=1.5, color='coral')
        ax3.set_xlabel('시간 (초)', fontweight='bold')
        ax3.set_ylabel('베타 파워', fontweight='bold')
        ax3.set_title('베타 밴드 파워 (13-30 Hz) - 채널 평균', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. 채널별 평균 알파 파워
        ax4 = fig.add_subplot(gs[2, 0])
        channel_alpha_mean = np.mean(self.features['alpha_power'], axis=0)
        ax4.bar(range(len(channel_alpha_mean)), channel_alpha_mean, color='steelblue', alpha=0.7)
        ax4.set_xlabel('채널', fontweight='bold')
        ax4.set_ylabel('평균 알파 파워', fontweight='bold')
        ax4.set_title('채널별 평균 알파 파워', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(self.processed_data['channels'])))
        ax4.set_xticklabels(self.processed_data['channels'], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. 채널별 평균 베타 파워
        ax5 = fig.add_subplot(gs[2, 1])
        channel_beta_mean = np.mean(self.features['beta_power'], axis=0)
        ax5.bar(range(len(channel_beta_mean)), channel_beta_mean, color='coral', alpha=0.7)
        ax5.set_xlabel('채널', fontweight='bold')
        ax5.set_ylabel('평균 베타 파워', fontweight='bold')
        ax5.set_title('채널별 평균 베타 파워', fontsize=12, fontweight='bold')
        ax5.set_xticks(range(len(self.processed_data['channels'])))
        ax5.set_xticklabels(self.processed_data['channels'], rotation=45, ha='right')
        ax5.grid(True, alpha=0.3, axis='y')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 시각화 저장: {save_path}")


def main():
    """메인 실행 함수"""
    print("\n" + "╔" + "="*88 + "╗")
    print("║" + " "*20 + "Redis EEG Processing Pipeline" + " "*39 + "║")
    print("║" + " "*15 + "Nature Communications 2025 논문 방법 적용" + " "*32 + "║")
    print("╚" + "="*88 + "╝\n")

    # 파이프라인 생성
    pipeline = RedisEEGPipeline()

    # Redis 연결
    if not pipeline.connect():
        return

    # 사용 가능한 시간 범위 확인
    time_range = pipeline.get_available_time_range()
    if time_range is None:
        return

    # 데이터 로드 (기본: 전체 데이터)
    # 특정 시간 범위를 원하면 start_time, end_time 파라미터 사용
    # 예: pipeline.load_data(start_time=100.0, end_time=200.0)
    if not pipeline.load_data():
        return

    # 전처리
    if not pipeline.preprocess(apply_ica=True):
        return

    # 세그먼트화
    if not pipeline.segment_data(window_size=1.0, step_size=0.125):
        return

    # 특징 추출
    if not pipeline.extract_features(alpha_band=(8, 13), beta_band=(13, 30)):
        return

    # 결과 시각화
    pipeline.visualize_results()

    # 데이터 저장
    output_file = pipeline.save_processed_data()

    # 최종 요약
    print("\n" + "="*90)
    print("✅ 파이프라인 실행 완료")
    print("="*90)
    print(f"\n다음 단계:")
    print(f"  1. 저장된 데이터 ({output_file})를 사용하여 모델 학습")
    print(f"  2. 새로운 데이터 수집 시 동일한 파이프라인 적용")
    print(f"  3. 실시간 분석을 위해 스트리밍 모드 구현 가능\n")


if __name__ == "__main__":
    main()
