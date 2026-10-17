# I checked the licences and videos of public robot datasets. Here's what I found.

*October 2026*

Robot learning runs on shared data. Most of us download a dataset from the Hub, check that it loads, and start training. I built an open-source checker, [rdq](https://github.com/s1g6r/rdq), to look at the parts we usually skip: whether we're allowed to use the data, and whether the camera footage is actually usable. Here is what it found.

## Licences: the tag on a dataset often isn't the owner's choice

LeRobot's `push_to_hub()` sets `license: apache-2.0` unless you change it, and writes a template dataset card. In the 3,000 most-downloaded datasets tagged LeRobot (October 2026):

- **56.5%** (51.9% of downloads) carry apache-2.0 on an unedited template card. That's most likely the upload tool's default, not a decision by whoever owns the data.
- **304** can be traced to one of 65 public robot datasets whose terms I checked at the source. **200 of them declare a different licence from the original**, mostly CC BY 4.0 data relabelled apache-2.0 or MIT. The data is still usable, but CC BY requires credit and a note of changes, and the new tag drops that on paper.
- **45 more** put a licence on data whose original release never licensed the data at all (only the code).

A tag on a copy can't grant rights the uploader doesn't hold, but people read the card as if it does. I opened [an issue on LeRobot](https://github.com/huggingface/lerobot/issues/4944) about the default, and a community fix that warns when the default is used is under review.

I also ran the check on every dataset on LeBoard, at the exact versions it pins: only 9 of 91 carry a licence that matches their source, and 4 have no dataset card at all.

## Video: problems you only see by watching every frame

rdq can now decode every frame of every episode and look for frozen or black stretches, blur, timing gaps, a camera that was moved between sessions, and episodes that repeat another episode's footage. I ran it on 20 datasets (2 official, 18 of the most-downloaded community ones; about 4,600 episodes) and checked every flag by looking at the frames. Three held up:

- a second camera that freezes for about 1.5 seconds at the start of six episodes while the main camera keeps moving;
- a dataset where the camera was moved after episode 19, so it mixes two viewpoints;
- a few 5-second episodes that show an empty scene with no task in them, nearly identical to each other.

I let each dataset's owners know. The third turned out to be on purpose: the owners replied that those clips were recorded with no bottle on the table, to teach the model not to reach for one that isn't there. So that one is a feature, not a defect, and it's a useful reminder of what a checker can and can't do: rdq can tell you an episode is unusual, not why. The other two are the kind of thing nobody notices until a policy underperforms.

I also want to be upfront about false alarms. Three came up on the way and were fixed before these results: duplicates matched in only one camera, slow play data that looked alike across episodes without being copies, and a moved camera first reported as blur. The rule rdq follows now: a copy must repeat the same movements at the same moments in every camera, and sharpness is judged against the same camera's own footage.

## Try it

```bash
pip install "rdq[video] @ git+https://github.com/s1g6r/rdq"
rdq check owner/dataset                 # licence, provenance and data checks
rdq check owner/dataset --video-full    # plus every frame of every episode
```

There's also a browser version of the licence check, no install: [huggingface.co/spaces/rs1g6r/rdq](https://huggingface.co/spaces/rs1g6r/rdq). The rules are written up in the [licence methodology](licence-methodology.md) and the [video methodology](raw-video.md), including the limits.

If you maintain a dataset, collect robot data, or buy it, I'd like to hear what checks you wish existed. If you send me a dataset or a sample delivery, I'll run it and send you the report. rdq is diligence, not legal advice.
