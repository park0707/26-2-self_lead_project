/**
 * vidio-health-trainer Web Dashboard Frontend
 * Interactive Skeleton Player & Biomechanical Defect Dashboard
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const serverStatusDot = document.getElementById("serverStatusDot");
  const serverStatusText = document.getElementById("serverStatusText");
  const exerciseSelect = document.getElementById("exerciseSelect");
  const viewSelect = document.getElementById("viewSelect");
  const strideSelect = document.getElementById("strideSelect");
  const dropzone = document.getElementById("dropzone");
  const videoFileInput = document.getElementById("videoFileInput");
  const selectedFileBadge = document.getElementById("selectedFileBadge");
  const startAnalyzeBtn = document.getElementById("startAnalyzeBtn");

  // Progress Card
  const progressCard = document.getElementById("progressCard");
  const progressBarFill = document.getElementById("progressBarFill");
  const progressPercentText = document.getElementById("progressPercentText");
  const progressStatusText = document.getElementById("progressStatusText");

  // Results Section
  const resultsSection = document.getElementById("resultsSection");
  const metricExercise = document.getElementById("metricExercise");
  const metricConfidence = document.getElementById("metricConfidence");
  const metricTotalReps = document.getElementById("metricTotalReps");
  const metricActivePeriod = document.getElementById("metricActivePeriod");
  const metricCleanReps = document.getElementById("metricCleanReps");
  const metricDuration = document.getElementById("metricDuration");
  const repTabsContainer = document.getElementById("repTabsContainer");
  const defectList = document.getElementById("defectList");

  // Video & Canvas Overlay Elements
  const canvas = document.getElementById("skeletonCanvas");
  const ctx = canvas.getContext("2d");
  const replayVideo = document.getElementById("replayVideo");
  const canvasContainer = document.getElementById("canvasContainer");
  const videoPlaceholder = document.getElementById("videoPlaceholder");
  const activeRepBadge = document.getElementById("activeRepBadge");
  const modeBtns = document.querySelectorAll(".mode-btn");
  const videoOpacitySlider = document.getElementById("videoOpacitySlider");
  const videoOpacityVal = document.getElementById("videoOpacityVal");
  const speedSelect = document.getElementById("speedSelect");
  const skeletonLegend = document.getElementById("skeletonLegend");
  const playPauseBtn = document.getElementById("playPauseBtn");
  const timelineSlider = document.getElementById("timelineSlider");
  const frameIndicator = document.getElementById("frameIndicator");

  // Application State
  let currentFile = null;
  let videoBlobUrl = null;
  let currentMode = "rtmp"; // 'rtmp' | 'mediapipe' | 'both' | 'skeleton'
  let analysisData = null;
  let activeRepIndex = 0;
  let activeRepData = null;
  let animationFrameId = null;
  let isPlaying = false;
  let currentPlaybackFrame = 0;
  let playbackFps = 30;

  // 1. Joint15 뼈대 연결선 (하위 호환 및 단순 뼈대 모드용)
  const SKELETON_EDGES_15 = [
    [0, 1],   // PELVIS - NECK
    [1, 2], [2, 3], // NECK - L_SHOULDER - L_ELBOW
    [3, 4],   // L_ELBOW - L_WRIST
    [1, 5], [5, 6], // NECK - R_SHOULDER - R_ELBOW
    [6, 7],   // R_ELBOW - R_WRIST
    [0, 8], [8, 9], // PELVIS - L_HIP - L_KNEE
    [9, 10],  // L_KNEE - L_ANKLE
    [0, 11], [11, 12], // PELVIS - R_HIP - R_KNEE
    [12, 13]  // R_KNEE - R_ANKLE
  ];

  // 2. RTMPose Halpe26 정밀 뼈대 엣지 (척추선 & 발 3각 지지면 포함)
  const RTMP_EDGES_26 = [
    [15, 13], [13, 11], [11, 19], // 좌측 하지 (발목-무릎-고관절-골반)
    [16, 14], [14, 12], [12, 19], // 우측 하지 (발목-무릎-고관절-골반)
    [17, 18], [18, 19],           // 척추 라인 (머리-경추, 경추-골반)
    [18, 5], [5, 7], [7, 9],      // 좌측 상지 (경추-어깨-팔꿈치-손목)
    [18, 6], [6, 8], [8, 10],     // 우측 상지 (경추-어깨-팔꿈치-손목)
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], // 안면/어깨 연결
    // 좌측 발 3각 지지면 (발목-엄지-새끼-뒤꿈치)
    [15, 20], [15, 22], [15, 24], [20, 22], [22, 24], [24, 20],
    // 우측 발 3각 지지면 (발목-엄지-새끼-뒤꿈치)
    [16, 21], [16, 23], [16, 25], [21, 23], [23, 25], [25, 21]
  ];

  // 3. MediaPipe 33 전신 랜드마크 엣지
  const MP_EDGES_33 = [
    // 안면
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
    // 상체 및 팔
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    // 손
    [15, 17], [15, 19], [15, 21], [17, 19], [16, 18], [16, 20], [16, 22], [18, 20],
    // 체간/골반
    [11, 23], [12, 24], [23, 24],
    // 하지 다리
    [23, 25], [25, 27], [24, 26], [26, 28],
    // 발 및 발목
    [27, 29], [27, 31], [29, 31], [28, 30], [28, 32], [30, 32]
  ];

  // =========================================================================
  // 1. 초기 서버 상태 및 메타데이터 로드
  // =========================================================================
  async function initApp() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("서버 응답 오류");
      const health = await res.json();

      serverStatusDot.classList.remove("offline");
      serverStatusText.textContent = `Online (33종 결함 모델 가동 중)`;

      // 운동 카탈로그 채우기
      exerciseSelect.innerHTML = `<option value="">[1단계 AI 자동 판별]</option>`;
      health.supported_exercises.forEach((ex) => {
        const opt = document.createElement("option");
        opt.value = ex.id;
        opt.textContent = `${ex.code.split("_")[0]}번: ${ex.name} (${ex.defects_count}개 결함)`;
        exerciseSelect.appendChild(opt);
      });

      // 공식 5대 시점 채우기
      viewSelect.innerHTML = "";
      health.camera_views.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = `${v.alias}: ${v.name}`;
        opt.textContent = `${v.alias} (${v.angle}) - ${v.name}`;
        if (v.code === "view3") opt.selected = true; // 정면 기본값
        viewSelect.appendChild(opt);
      });
    } catch (err) {
      console.error("초기화 실패:", err);
      serverStatusDot.classList.add("offline");
      serverStatusText.textContent = "Offline (서버 연결 불가)";
    }
  }

  initApp();

  // =========================================================================
  // 2. 파일 드래그 앤 드롭 이벤트 핸들러
  // =========================================================================
  dropzone.addEventListener("click", () => videoFileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  videoFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    const validExtensions = [".mp4", ".mov", ".avi", ".webm", ".mkv"];
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!validExtensions.includes(ext)) {
      alert("지원하지 않는 동영상 포맷입니다: " + ext);
      return;
    }

    currentFile = file;
    selectedFileBadge.textContent = `✔ 선택된 영상: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    selectedFileBadge.style.display = "inline-block";
    startAnalyzeBtn.disabled = false;

    // 브라우저 로컬 Blob URL 즉시 연결 (서버 대기 및 네트워크 지연 0)
    if (videoBlobUrl) {
      URL.revokeObjectURL(videoBlobUrl);
    }
    videoBlobUrl = URL.createObjectURL(file);
    replayVideo.src = videoBlobUrl;
    replayVideo.load();
    if (videoPlaceholder) {
      videoPlaceholder.classList.add("hidden");
    }
  }

  // =========================================================================
  // 3. 비디오 분석 요청 및 실시간 WebSocket 진행률
  // =========================================================================
  startAnalyzeBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    // UI 상태 초기화
    startAnalyzeBtn.disabled = true;
    progressCard.style.display = "block";
    resultsSection.style.display = "none";
    updateProgress(5, "작업 준비 중...");

    const channelId = `ch_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProto}//${window.location.host}/api/ws/analyze/${channelId}`;

    let socket = null;
    try {
      socket = new WebSocket(wsUrl);
      socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "progress") {
            updateProgress(msg.progress_percent, msg.message);
          }
        } catch (e) {
          console.warn("WS 파싱 에러:", e);
        }
      };
    } catch (e) {
      console.warn("WebSocket 연결 생략:", e);
    }

    // FormData 생성
    const formData = new FormData();
    formData.append("video_file", currentFile);
    if (exerciseSelect.value !== "") {
      formData.append("exercise_id", exerciseSelect.value);
    }
    formData.append("camera_view", viewSelect.value);
    formData.append("sample_stride", strideSelect ? strideSelect.value : "2");
    formData.append("channel_id", channelId);

    try {
      const response = await fetch("/api/analyze/video", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "분석 요청 실패");
      }

      analysisData = await response.json();
      updateProgress(100, "분석 완료!");
      setTimeout(() => {
        progressCard.style.display = "none";
        renderResults(analysisData);
        startAnalyzeBtn.disabled = false;
      }, 500);
    } catch (error) {
      console.error("분석 에러:", error);
      alert("분석 실패: " + error.message);
      progressCard.style.display = "none";
      startAnalyzeBtn.disabled = false;
    } finally {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    }
  });

  function updateProgress(percent, message) {
    const p = Math.min(100, Math.max(0, percent));
    progressBarFill.style.width = `${p}%`;
    progressPercentText.textContent = `${p.toFixed(0)}%`;
    progressStatusText.textContent = message;
  }

  // =========================================================================
  // 4. 분석 결과 렌더링 및 반복별 결함 카드 생성
  // =========================================================================
  function renderResults(data) {
    resultsSection.style.display = "flex";

    // 1. 상단 요약 카드 채우기
    metricExercise.textContent = data.exercise.exercise_name;
    metricConfidence.textContent = `확신도: ${(data.exercise.confidence * 100).toFixed(1)}%`;
    metricTotalReps.textContent = `${data.total_reps}회`;
    if (data.active_period) {
      metricActivePeriod.textContent = `활성 구간: ${data.active_period.duration_sec}초 (F${data.active_period.start_frame}~F${data.active_period.end_frame})`;
    } else {
      metricActivePeriod.textContent = "1회 반복 단위 자동 분할";
    }

    const cleanCount = data.reps.filter((r) => r.is_clean).length;
    metricCleanReps.textContent = `${cleanCount}회 (${((cleanCount / (data.total_reps || 1)) * 100).toFixed(0)}%)`;
    if (data.video_info && data.video_info.effective_fps) {
      const strideVal = data.video_info.sample_stride || 2;
      const modeBadge = strideVal > 1 ? ` (⚡2배속 ${data.video_info.effective_fps}FPS)` : ` (30FPS 전수)`;
      metricDuration.textContent = `${data.total_processing_time_sec}초${modeBadge}`;
    } else {
      metricDuration.textContent = `${data.total_processing_time_sec}초`;
    }

    // 2. 반복 회차 탭 생성
    repTabsContainer.innerHTML = "";
    data.reps.forEach((rep, idx) => {
      const tab = document.createElement("button");
      tab.className = `rep-tab ${idx === 0 ? "active" : ""}`;
      const badgeClass = rep.is_clean ? "clean" : "defect";
      tab.innerHTML = `<span class="tab-badge ${badgeClass}"></span> Rep #${rep.rep_index}`;
      tab.addEventListener("click", () => selectRepetition(idx));
      repTabsContainer.appendChild(tab);
    });

    // 3. 3단계 세트 종합 맞춤 처방전 렌더링
    let setSummaryEl = document.getElementById("setSummaryContainer");
    if (!setSummaryEl) {
      setSummaryEl = document.createElement("div");
      setSummaryEl.id = "setSummaryContainer";
      setSummaryEl.className = "set-summary-container";
      const cards = document.querySelectorAll(".results-card");
      if (cards.length >= 2) {
        cards[1].appendChild(setSummaryEl);
      }
    }
    if (data.set_summary && setSummaryEl) {
      const sm = data.set_summary;
      const drillPills = (sm.recommended_drills || [])
        .map((d) => `<span class="drill-pill">⚡ ${d}</span>`)
        .join(" ");
      setSummaryEl.style.display = "block";
      setSummaryEl.innerHTML = `
        <div class="set-summary-header">
          <div class="set-summary-title">
            <span>📋</span> ${sm.summary_title}
          </div>
          <span class="set-clean-rate">완전 정상률: ${sm.clean_ratio_percent}% (${sm.clean_reps}/${sm.total_reps}회)</span>
        </div>
        <div class="set-prescription-box">
          ${sm.overall_prescription}
        </div>
        <div class="set-drills-list">
          <span class="drill-label">추천 보강 드릴:</span>
          ${drillPills}
        </div>
      `;
    }

    // 4. 첫 번째 반복 선택 및 바인딩
    selectRepetition(0);
  }

  function selectRepetition(index) {
    if (!analysisData || !analysisData.reps[index]) return;
    activeRepIndex = index;

    // 탭 활성화 UI 갱신
    const tabs = repTabsContainer.querySelectorAll(".rep-tab");
    tabs.forEach((t, i) => {
      t.classList.toggle("active", i === index);
    });

    const rep = analysisData.reps[index];

    // 결함 리스트 렌더링
    defectList.innerHTML = "";

    // 3단계 회차 최우선 행동 지침 배너 렌더링 (Stage 3 Actionable Cueing)
    if (rep.primary_feedback) {
      const banner = document.createElement("div");
      banner.className = `coaching-primary-banner ${rep.is_clean ? "clean" : "defect"}`;
      banner.innerHTML = `
        <span class="banner-badge">${rep.is_clean ? "✨ 폼 유지 (Clean)" : "🎯 최우선 교정 지침"}</span>
        <div class="banner-cue">${rep.primary_feedback}</div>
        ${rep.primary_detail ? `<div class="banner-detail">${rep.primary_detail}</div>` : ""}
        ${rep.causal_chain_title ? `<div class="banner-chain">🔗 <b>연쇄 인과 분석:</b> ${rep.causal_chain_title}</div>` : ""}
        ${rep.view_coaching_tip ? `<div class="banner-tip">💡 <b>시점 안내:</b> ${rep.view_coaching_tip}</div>` : ""}
      `;
      defectList.appendChild(banner);
    }

    rep.defects.forEach((df) => {
      const item = document.createElement("div");
      let stateClass = "clean";
      let stateText = "정상";
      if (df.state === 1) {
        stateClass = "defect";
        stateText = "결함";
      } else if (df.state === 2) {
        stateClass = "caution";
        stateText = "주의/보류";
      }

      const probPercent = (df.probability * 100).toFixed(1);

      item.className = `defect-item ${stateClass}`;
      item.innerHTML = `
        <div class="defect-header">
          <div class="defect-code-name">
            <span class="code-pill">${df.defect_code}</span>
            <span class="defect-title">${df.defect_name}</span>
          </div>
          <span class="state-badge ${stateClass}">${stateText}</span>
        </div>
        <div class="prob-bar-container">
          <div class="prob-bar-bg">
            <div class="prob-bar-fill ${stateClass}" style="width: ${probPercent}%"></div>
          </div>
          <span class="prob-label">${probPercent}%</span>
        </div>
        <div class="feedback-box">
          <div class="feedback-title">정상 동작을 위한 교정 조치</div>
          <div style="font-weight: 500; line-height: 1.45;">${df.corrective_action || df.feedback}</div>
        </div>
      `;
      defectList.appendChild(item);
    });

    // 4. 반복 데이터 바인딩 및 플레이어 초기화
    activeRepData = rep;
    activeRepBadge.textContent = `반복 #${rep.rep_index} (${rep.start_sec}s ~ ${rep.end_sec}s, ${rep.duration_sec}초)`;

    if (videoBlobUrl && replayVideo) {
      replayVideo.currentTime = rep.start_sec || 0;
    }

    currentPlaybackFrame = 0;
    timelineSlider.max = 63;
    timelineSlider.value = 0;
    drawCurrentFrame(0);

    playPauseBtn.disabled = false;
    timelineSlider.disabled = false;
  }

  // =========================================================================
  // 5. 모드 전환 및 비디오 투명도 / 재생 속도 이벤트 핸들러
  // =========================================================================
  modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      modeBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentMode = btn.dataset.mode;

      // 비디오 표시 제어
      if (currentMode === "skeleton") {
        replayVideo.style.opacity = "0";
      } else {
        replayVideo.style.opacity = videoOpacitySlider.value;
      }

      // 범례 표시 제어
      if (skeletonLegend) {
        skeletonLegend.style.display = currentMode === "skeleton" ? "none" : "flex";
      }

      drawCurrentFrame(currentPlaybackFrame);
    });
  });

  videoOpacitySlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    videoOpacityVal.textContent = `${Math.round(val * 100)}%`;
    if (currentMode !== "skeleton") {
      replayVideo.style.opacity = val;
    }
  });

  speedSelect.addEventListener("change", (e) => {
    const rate = parseFloat(e.target.value);
    replayVideo.playbackRate = rate;
  });

  // =========================================================================
  // 6. 비디오 렌더링 영역(Letterbox / Pillarbox) 좌표 매핑
  // =========================================================================
  function getVideoRenderBox() {
    const cW = canvasContainer.clientWidth || 480;
    const cH = canvasContainer.clientHeight || 270;

    // Retina / HiDPI 선명도 보정을 위한 캔버스 내부 해상도 동기화
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== cW * dpr || canvas.height !== cH * dpr) {
      canvas.width = cW * dpr;
      canvas.height = cH * dpr;
    }
    ctx.resetTransform();
    ctx.scale(dpr, dpr);

    const vW = replayVideo.videoWidth;
    const vH = replayVideo.videoHeight;

    if (!vW || !vH || currentMode === "skeleton" || !videoBlobUrl) {
      // 비디오가 없거나 단순 뼈대 모드인 경우 캔버스 중앙 80% 영역에 렌더링
      const size = Math.min(cW, cH);
      return {
        left: (cW - size * 0.9) / 2,
        top: (cH - size * 0.9) / 2,
        width: size * 0.9,
        height: size * 0.9,
        isCentered: true
      };
    }

    const videoRatio = vW / vH;
    const containerRatio = cW / cH;
    let rW, rH, left, top;

    if (containerRatio > videoRatio) {
      // Pillarbox (좌우 여백)
      rH = cH;
      rW = cH * videoRatio;
      left = (cW - rW) / 2;
      top = 0;
    } else {
      // Letterbox (상하 여백)
      rW = cW;
      rH = cW / videoRatio;
      left = 0;
      top = (cH - rH) / 2;
    }

    return { left, top, width: rW, height: rH, isCentered: false };
  }

  // =========================================================================
  // 7. 실시간 64-Frame 스켈레톤 & 비디오 동기화 오버레이 드로잉
  // =========================================================================
  function clearCanvas() {
    const cW = canvasContainer.clientWidth || 480;
    const cH = canvasContainer.clientHeight || 270;

    if (currentMode === "skeleton") {
      ctx.fillStyle = "#090d16";
      ctx.fillRect(0, 0, cW, cH);
    } else {
      ctx.clearRect(0, 0, cW, cH);
    }
  }

  function drawCurrentFrame(frameIdx) {
    if (!activeRepData) return;
    clearCanvas();

    const box = getVideoRenderBox();
    const rep = activeRepData;

    // 현재 모드에 따라 뼈대 렌더링
    if (currentMode === "rtmp") {
      if (rep.overlay_rtmp && rep.overlay_rtmp[frameIdx]) {
        drawRTMPoseSkeleton(rep.overlay_rtmp[frameIdx], box, false);
      } else if (rep.keypoints_preview) {
        drawFallbackJoint15(rep.keypoints_preview[frameIdx], box);
      }
    } else if (currentMode === "mediapipe") {
      if (rep.overlay_mp && rep.overlay_mp[frameIdx]) {
        drawMediaPipeSkeleton(rep.overlay_mp[frameIdx], box, false);
      } else if (rep.keypoints_preview) {
        drawFallbackJoint15(rep.keypoints_preview[frameIdx], box);
      }
    } else if (currentMode === "both") {
      // 듀얼 비교: MediaPipe를 배경에 은은하게, RTMPose를 전면에 선명하게
      if (rep.overlay_mp && rep.overlay_mp[frameIdx]) {
        drawMediaPipeSkeleton(rep.overlay_mp[frameIdx], box, true);
      }
      if (rep.overlay_rtmp && rep.overlay_rtmp[frameIdx]) {
        drawRTMPoseSkeleton(rep.overlay_rtmp[frameIdx], box, false);
      }
    } else if (currentMode === "spinepose") {
      // SpinePose 모드: MediaPipe 윤곽을 은은하게 깔고 9대 정밀 척추선을 전면에 강조
      if (rep.overlay_mp && rep.overlay_mp[frameIdx]) {
        drawMediaPipeSkeleton(rep.overlay_mp[frameIdx], box, true);
      }
      if (rep.overlay_spine && rep.overlay_spine[frameIdx]) {
        drawSpinePoseSkeleton(rep.overlay_spine[frameIdx], box);
      } else if (rep.overlay_rtmp && rep.overlay_rtmp[frameIdx]) {
        drawRTMPoseSkeleton(rep.overlay_rtmp[frameIdx], box, false);
      }
    } else {
      // 단순 뼈대 모드
      if (rep.keypoints_preview && rep.keypoints_preview[frameIdx]) {
        drawFallbackJoint15(rep.keypoints_preview[frameIdx], box);
      }
    }

    // 시간 및 프레임 표시 갱신
    let timeText = "";
    if (rep.start_sec !== undefined && rep.duration_sec) {
      const curSec = rep.start_sec + (frameIdx / 63.0) * rep.duration_sec;
      timeText = ` (${curSec.toFixed(2)}s)`;
    }
    frameIndicator.textContent = `${frameIdx + 1} / 64F${timeText}`;
  }

  // 7-1. RTMPose Halpe26 드로잉 (발 3각 지지면 & 척추 특화)
  function drawRTMPoseSkeleton(kps26, box, isSecondary) {
    const pts = kps26.map(([nx, ny]) => [
      box.left + nx * box.width,
      box.top + ny * box.height
    ]);

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // 1. 뼈대 연결선 드로잉
    RTMP_EDGES_26.forEach(([i, j]) => {
      const p1 = pts[i];
      const p2 = pts[j];
      if (!p1 || !p2 || isNaN(p1[0]) || isNaN(p2[0])) return;

      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);

      // 부위별 차별화 컬러 (발 3각 지지면: 골드/옐로우, 척추: 시안, 사지: 에메랄드)
      if (i >= 20 || j >= 20) {
        ctx.strokeStyle = "#fbbf24"; // 발 3각 지지면 (골드)
        ctx.lineWidth = isSecondary ? 2 : 3.5;
        ctx.shadowColor = "#fbbf24";
        ctx.shadowBlur = isSecondary ? 2 : 6;
      } else if (i === 17 || i === 18 || i === 19 || j === 17 || j === 18 || j === 19) {
        ctx.strokeStyle = "#38bdf8"; // 척추 및 경추 라인 (스카이블루)
        ctx.lineWidth = isSecondary ? 2.5 : 4;
        ctx.shadowColor = "#38bdf8";
        ctx.shadowBlur = isSecondary ? 2 : 8;
      } else {
        ctx.strokeStyle = "#10b981"; // 사지 팔다리 (에메랄드)
        ctx.lineWidth = isSecondary ? 2 : 3;
        ctx.shadowColor = "#10b981";
        ctx.shadowBlur = isSecondary ? 1 : 5;
      }
      ctx.stroke();
    });

    // 2. 관절 포인트 드로잉
    pts.forEach(([x, y], idx) => {
      if (isNaN(x) || isNaN(y)) return;
      ctx.beginPath();
      const isFoot = idx >= 20;
      const isCenter = idx === 17 || idx === 18 || idx === 19;
      const radius = isFoot ? 4 : (isCenter ? 5 : 3.5);

      ctx.arc(x, y, isSecondary ? radius * 0.8 : radius, 0, 2 * Math.PI);
      ctx.fillStyle = isFoot ? "#f59e0b" : (isCenter ? "#ec4899" : "#10b981");
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    ctx.restore();
  }

  // 7-2. MediaPipe 33 전신 드로잉 (전신 실루엣 특화)
  function drawMediaPipeSkeleton(kps33, box, isSecondary) {
    const pts = kps33.map(([nx, ny]) => [
      box.left + nx * box.width,
      box.top + ny * box.height
    ]);

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (isSecondary) {
      ctx.globalAlpha = 0.65;
    }

    // 1. 뼈대 연결선 드로잉
    MP_EDGES_33.forEach(([i, j]) => {
      const p1 = pts[i];
      const p2 = pts[j];
      if (!p1 || !p2 || isNaN(p1[0]) || isNaN(p2[0])) return;

      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);

      if (i >= 27 || j >= 27) {
        ctx.strokeStyle = "#ec4899"; // 발끝 (핑크)
        ctx.lineWidth = isSecondary ? 1.8 : 2.5;
      } else if (i <= 10 || j <= 10) {
        ctx.strokeStyle = "#a78bfa"; // 얼굴 (퍼플)
        ctx.lineWidth = isSecondary ? 1.2 : 1.8;
      } else if (i === 11 || i === 12 || i === 23 || i === 24 || j === 11 || j === 12 || j === 23 || j === 24) {
        ctx.strokeStyle = "#00e5ff"; // 체간 몸통 (시안)
        ctx.lineWidth = isSecondary ? 2.5 : 3.5;
      } else {
        ctx.strokeStyle = "#818cf8"; // 팔다리 (인디고)
        ctx.lineWidth = isSecondary ? 2 : 2.8;
      }
      ctx.stroke();
    });

    // 2. 관절 포인트 드로잉
    pts.forEach(([x, y], idx) => {
      if (isNaN(x) || isNaN(y)) return;
      ctx.beginPath();
      const r = idx === 11 || idx === 12 || idx === 23 || idx === 24 ? 4 : 3;
      ctx.arc(x, y, isSecondary ? r * 0.7 : r, 0, 2 * Math.PI);
      ctx.fillStyle = idx >= 27 ? "#ec4899" : (idx <= 10 ? "#c4b5fd" : "#00e5ff");
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    ctx.restore();
  }

  // 7-3. SpinePose 9대 척추 분절선 특화 드로잉
  const SPINE_EDGES_9 = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]
  ];
  const SPINE_LABELS_9 = ["S1", "L5", "L3", "L1", "T8", "T3", "C7", "C4", "C1"];

  function drawSpinePoseSkeleton(kps9, box) {
    const pts = kps9.map(([nx, ny]) => [
      box.left + nx * box.width,
      box.top + ny * box.height
    ]);

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // 1. 척추 체인 연결선
    SPINE_EDGES_9.forEach(([i, j]) => {
      const p1 = pts[i];
      const p2 = pts[j];
      if (!p1 || !p2 || isNaN(p1[0]) || isNaN(p2[0])) return;

      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);

      // 요추 구간(0~3: S1-L5-L3-L1)은 버트윙크 핵심 부위로 마젠타 네온
      if (i <= 2) {
        ctx.strokeStyle = "#ff007a"; // 요추 핵심선 (네온 마젠타)
        ctx.lineWidth = 5.0;
        ctx.shadowColor = "#ff007a";
        ctx.shadowBlur = 10;
      } else if (i <= 5) {
        ctx.strokeStyle = "#a855f7"; // 흉추선 (퍼플)
        ctx.lineWidth = 4.2;
        ctx.shadowColor = "#a855f7";
        ctx.shadowBlur = 8;
      } else {
        ctx.strokeStyle = "#38bdf8"; // 경추선 (스카이블루)
        ctx.lineWidth = 3.5;
        ctx.shadowColor = "#38bdf8";
        ctx.shadowBlur = 6;
      }
      ctx.stroke();
    });

    // 2. 9대 척추 마디 포인트 및 라벨 드로잉
    pts.forEach(([x, y], idx) => {
      if (isNaN(x) || isNaN(y)) return;
      ctx.beginPath();
      const isLumbar = idx <= 3; // S1, L5, L3, L1
      const radius = isLumbar ? 5.5 : 4.5;

      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isLumbar ? "#ff007a" : (idx <= 5 ? "#c084fc" : "#38bdf8");
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.0;
      ctx.stroke();

      // 마디 라벨 표시 (S1, L5, L1, T8, C7 등)
      ctx.font = "bold 9px 'Pretendard', sans-serif";
      ctx.fillStyle = "#ffffff";
      ctx.shadowColor = "#000000";
      ctx.shadowBlur = 4;
      ctx.fillText(SPINE_LABELS_9[idx] || "", x + 8, y + 3);
    });

    ctx.restore();
  }

  // 7-4. 순수 뼈대 / 폴백 15관절 드로잉
  function drawFallbackJoint15(kps15, box) {
    const W = box.width;
    const H = box.height;
    const pts = kps15.map((pt) => [
      box.left + W * 0.5 + pt[0] * (W * 0.45),
      box.top + H * 0.5 + pt[1] * (H * 0.45)
    ]);

    ctx.save();
    ctx.lineWidth = 3;
    SKELETON_EDGES_15.forEach(([i, j]) => {
      if (!pts[i] || !pts[j]) return;
      ctx.beginPath();
      ctx.moveTo(pts[i][0], pts[i][1]);
      ctx.lineTo(pts[j][0], pts[j][1]);
      if (i <= 1 && j <= 1) {
        ctx.strokeStyle = "#38bdf8";
      } else if (i >= 8 || j >= 8) {
        ctx.strokeStyle = "#34d399";
      } else {
        ctx.strokeStyle = "#818cf8";
      }
      ctx.stroke();
    });

    pts.forEach(([x, y], idx) => {
      ctx.beginPath();
      ctx.arc(x, y, idx === 0 || idx === 1 ? 5 : 4, 0, 2 * Math.PI);
      ctx.fillStyle = idx === 0 ? "#f43f5e" : "#fbbf24";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
    ctx.restore();
  }

  // =========================================================================
  // 8. 비디오-스켈레톤 완전 동기화 재생 루프
  // =========================================================================
  let lastTime = 0;

  function animationLoop(timestamp) {
    if (!isPlaying) return;

    const rep = activeRepData;
    const hasVideo = videoBlobUrl && replayVideo && !replayVideo.paused && currentMode !== "skeleton";

    if (hasVideo && rep) {
      // 1. 비디오 시간 동기화 모드
      const startS = rep.start_sec || 0;
      const endS = rep.end_sec || (startS + rep.duration_sec);
      const span = Math.max(0.1, endS - startS);
      const curTime = replayVideo.currentTime;

      if (curTime >= endS || curTime < startS - 0.2) {
        // 구간 반복 루프
        replayVideo.currentTime = startS;
      } else {
        const progress = Math.min(1.0, Math.max(0.0, (curTime - startS) / span));
        currentPlaybackFrame = Math.min(63, Math.floor(progress * 64));
        timelineSlider.value = currentPlaybackFrame;
        drawCurrentFrame(currentPlaybackFrame);
      }
    } else {
      // 2. 가상 타이머 독립 재생 모드 (비디오 미사용 또는 뼈대 모드)
      const currentRate = parseFloat(speedSelect.value) || 0.5;
      const targetFps = 30 * currentRate;
      const frameInterval = 1000 / targetFps;

      if (timestamp - lastTime >= frameInterval) {
        lastTime = timestamp;
        currentPlaybackFrame = (currentPlaybackFrame + 1) % 64;
        timelineSlider.value = currentPlaybackFrame;
        drawCurrentFrame(currentPlaybackFrame);
      }
    }

    animationFrameId = requestAnimationFrame(animationLoop);
  }

  playPauseBtn.addEventListener("click", () => {
    if (!activeRepData) return;
    isPlaying = !isPlaying;
    playPauseBtn.textContent = isPlaying ? "⏸ 일시정지" : "▶ 재생";

    if (isPlaying) {
      lastTime = performance.now();
      if (videoBlobUrl && currentMode !== "skeleton") {
        const rep = activeRepData;
        const startS = rep.start_sec || 0;
        const endS = rep.end_sec || (startS + rep.duration_sec);
        if (replayVideo.currentTime < startS || replayVideo.currentTime >= endS) {
          replayVideo.currentTime = startS;
        }
        replayVideo.play().catch((e) => console.warn("Video play interrupted:", e));
      }
      animationFrameId = requestAnimationFrame(animationLoop);
    } else {
      if (videoBlobUrl) {
        replayVideo.pause();
      }
      cancelAnimationFrame(animationFrameId);
    }
  });

  timelineSlider.addEventListener("input", (e) => {
    currentPlaybackFrame = parseInt(e.target.value, 10);
    if (activeRepData && videoBlobUrl && currentMode !== "skeleton") {
      const rep = activeRepData;
      const startS = rep.start_sec || 0;
      const endS = rep.end_sec || (startS + rep.duration_sec);
      const span = Math.max(0.1, endS - startS);
      replayVideo.currentTime = startS + (currentPlaybackFrame / 63.0) * span;
    }
    drawCurrentFrame(currentPlaybackFrame);
  });
});
