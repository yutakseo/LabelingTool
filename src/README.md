# 라벨러 모듈 구조

실행은 프로젝트 루트에서 `python main.py` 또는 기존 `start.bat` / `start.command`를 사용합니다. 명령행 인자도 그대로 지원합니다 (`python main.py --help`).

기본 입력 이미지, 초기 마스크, 출력 경로와 클래스·브러시·줌 설정은 **config.py**에서 수정합니다. 기존 Windows 기본 경로를 유지했으므로 다른 컴퓨터에서는 해당 경로를 변경하거나 명령행 인자로 지정하세요.

| 모듈 | 담당 기능 |
| --- | --- |
| `config.py` | 기본 경로, 클래스 정의, 편집 설정 |
| `cli.py` | 명령행 인자와 실행 경로 결정 |
| `classes.py` | 클래스 검증, 색상, 기존 클래스 메타데이터 복원 |
| `images.py` | 이미지 탐색과 RGB 로딩 |
| `masks.py` | 마스크 로딩·합성·미리보기, 영역 채우기 |
| `video.py` | 비디오 정보, FPS 선택, 프레임 추출 |
| `dataset.py` | 데이터셋 준비, 이미지/마스크 매칭, 메타데이터 저장 |
| `history.py` | 레이어별 undo/redo 스냅샷 |
| `input_events.py` | Qt 키보드·마우스·포커스 처리 |
| `types.py` | 콜백과 프레임 데이터 공통 타입 |
| `app.py` | napari UI, 편집 세션, 캐시·자동저장, 실행/종료 |

`main.py`는 `src.app.main()`을 호출하는 진입점입니다. UI 콜백은 세션 상태를 공유하므로 `app.py`에 함께 두었습니다. `CLASS_DEFINITIONS`는 `config.py`의 같은 딕셔너리를 공유하며, 실행 중 클래스 추가·복원이 모든 모듈에 반영됩니다.
