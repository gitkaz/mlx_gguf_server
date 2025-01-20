# このドキュメントは古く、現在の実装にあっていないため、更新の必要があります。

KV Cache の利用について

##　[前書き]

### [KV Cache について]
このプログラムでは、MLXでのChat Completionの場合に、KV Cacheを利用することができます。KV Cacheを使うと、次のようなメリットがあります。

* 高速なPrompt Eval
  + MLXは、NVidiaのGPUに比べて、Prompt Evaluationに時間がかかるという課題があります。特に長いPromptになるほど、Prompt Evalにかかる時間は顕著に延びていきます。チャットの場合、LLMとの会話が長くなるにつれて、徐々にレスポンス速度が遅くなってしまいます。
  + これは、プロンプトに、チャットのそれまでのメッセージヒストリをすべて載せているためです。つまり、実際にはLLMに対して過去のチャット履歴を渡しながら毎回「新規の」チャットをしていることになります。
  + KV Cacheを利用すると、過去のテキスト生成で利用した計算結果を再利用します。それによって、LLMが改めて計算が必要な部分は、あくまで新規に受け取ったメッセージのみに削減できます。したがって、Prompt Evalにかかる時間は、いつも、ユーザから受け取った最新のメッセージの長さのみに比例します。チャットのメッセージヒストリの量に影響されません。

MLXのmlx_lm.generate では、以前からKV Cacheが実装されていましたが、私が確認した限り、それは、一回のPromptの中だけで完結しており、チャットのような複数回のメッセージのやり取りに利用するものではなさそうでした。私は、mlx_lm.utils.generate_step にほんのわずかにコードを追加しただけで、KV Cacheをチャットのような継続的なやりとりで何度も利用・更新できるようにできました。mlxおよびmlx-examples の開発者の皆様に深く感謝します。（そのコードについては、llm_process/generate_stream.py 内のext_generate_step 関数を参照してください）

KV Cache の有無によるPromptの違いを図で示したもの

| Text Generation without KV Cache | Text Generation with KV Cache |
----|----
| (SVG) | (SVG)| 


### [KV Cache の有無によるテキスト生成速度の違い]
`mlx-community/gemma-2-27b-it-8bit` を使って、KV Cache の有無でどれくらい生成速度が違うかを動画にしました。
1. 大きな記事(これ)を読み込ませる
2. つぎのターンで「summarize the article」と要求する
3. サマリーした文章の出力開始までの時間差を計測


### [計測結果]
| | KV Cache を使用しない | KV Cache を使用する |
----|----|----
| | (動画) | (動画) |
|1ターン目| prompt_tokens: 4487, prompt_eval_time: 14.320998374954797 sec | prompt_tokens: 4487, prompt_eval_time: 13.99744424992241 sec |
|2ターン目| prompt_tokens: 4503, prompt_eval_time: 14.016152999945916 sec | prompt_tokens: 14, prompt_eval_time: 0.8610775420675054 sec |

１ターン目ではKV Cacheがないからどちらも同じだが、２ターン目で、KV Cacheが有効なために prompt eval token 数に大きな差があり、その結果 XX 倍、応答が高速化している。

## [注意]
KV Cache を利用した場合と、利用しない場合では出力は同一とはならない場合があります。（たとえ temperature = 0 であっても）



以下に、mlx_gguf_sever でのKV Cacheの利用方法について説明します。

## [前提条件]
* 対象はMLX フォーマットのモデルであること
* API Endpointは `/v1/chat/completions` であること
* POST に `kv_cache_session_id: 数値` が含まれていること
以上が条件です。

## [動作]
上記の前提条件に該当しているアクセスでは、テキスト生成時に以下を行います。
1.  すでにキャッシュがあるかどうかを判定
  + KV Cacheは内部では"kv_cache_session_id" に設定された数値を識別子として管理しています。
2. キャッシュが見当たらない場合は、通常通りに全てのチャットメッセージをプロンプトに入れて、テキストを生成します。同時に、指定されたkv_cache_session_idで、KV Cacheを生成し、メモリ内に保存します
3. 1のタイミングでキャッシュが見つかった場合は、そのキャッシュを利用します。この際は、テキスト生成時のプロンプトは、messages の末尾、つまりユーザからの最新の入力のみを用います。

## [KV Cacheの管理]
KV Cache の確認と削除には、以下のAPI 円dポイントが利用できます。

* /v1/internal/model/kv_cache/info
  + メソッド: GET
  + 対象モデルの現在のKV CacheのIDと、その容量(Byte)を応答しますが、現在、容量は(おそらく)正しい数値になっていません。

* /v1/internal/model/kv_cache/remove_cache
  + メソッド: POST
  + POSTパラメータ: {session_id: セッションID(数値)}
  + session_id で指定した KV Cacheを削除します。session_id で指定してるため、削除可能なKV Cacheは一度に一つです。

* /v1/internal/model/kv_cache/remove_old_caches
  + メソッド: POST
  + POSTパラメータ: {seconds: 秒数(数値)}
  + 現時点から seconds で指定した秒数より前が最終更新時刻であるKV Cacheを一度で複数削除します。


