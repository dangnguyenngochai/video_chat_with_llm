from pytubefix import YouTube

import gradio as gr

import gradio_app
from gradio_app.theme import minigptlv_style, custom_css,text_css

from preprocess import (
    extract_transcription,
    extract_frames
    )

from main import process_videos
import sys

sys.path.append('/embedding')
sys.path.append('/retrieval_generation')

def get_video_url(url):
    # get video id from url
    video_id=url.split('v=')[-1].split('&')[0]
    # Create a YouTube object
    youtube = YouTube(url)
    # Get the best available video stream
    video_stream = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    # Download the video to the workspace folder
    print('Downloading video')
    video_stream.download(output_path="workspace",filename=f"{video_id}.mp4")
    print('Video downloaded successfully')
    return f"data/video/{video_id}.mp4"

def download_video(youtube_url, download_finish):
    video_id=youtube_url.split('v=')[-1].split('&')[0]
    # Create a YouTube object
    youtube = YouTube(youtube_url)
    # Get the best available video stream
    video_stream = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    # if has_subtitles:
    # Download the video to the workspace folder
    print('Downloading video')
    video_stream.download(output_path="workspace",filename=f"{video_id}.mp4")
    print('Video downloaded successfully')
    processed_video_path= f"data/video/{video_id}.mp4"
    download_finish = gr.State(value=True)
    return processed_video_path, download_finish


def run_demo_youtube_video(youtube_url, query):
    try:
        video_file_path = get_video_url(youtube_url)
        process_videos(video_file_path, query)
    except Exception as ex:
        print('Keep going !!! Almost there')

def run_demo_local_video(video_file_path, query):
    try:
        process_videos(video_file_path, query)
    except Exception as ex:
        print('Keep going !!! Almost there')

title = """<h1 align="center">MiniGPT4-video 🎞️🍿</h1>"""
description = """<h5>This is the demo of MiniGPT4-video Model.</h5>"""
project_page = """<p><a href='https://vision-cair.github.io/MiniGPT4-video/'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p>"""
code_link="""<p><a href='https://github.com/Vision-CAIR/MiniGPT4-video'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p>"""
paper_link="""<p><a href=''><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>"""
video_path=""
with gr.Blocks(title="Video Chat Prototype 🎞️🍿",css=text_css ) as demo :
    # with gr.Row():
    #     with gr.Column(scale=2):
    gr.Markdown(title)
    gr.Markdown(description)
        # gr.Image("repo_imgs/Designer_2_new.jpeg",scale=1,show_download_button=False,show_label=False)
    # with gr.Row():
    #     gr.Markdown(project_page)
    #     gr.Markdown(code_link)
    #     gr.Markdown(paper_link)
        
    with gr.Tab("Local videos"):
        # local_interface=gr.Interface(
        #     fn=gradio_demo_local,
        #     inputs=[gr.Video(sources=["upload"]),gr.Checkbox(label='Use subtitles'),gr.Textbox(label="Write any Question")],
        #     outputs=["text",
        #             ],
            
        #     # title="<h2>Local videos</h2>",
        #     description="Upload your videos with length from one to two minutes",
        #     examples=[
        #         ["example_videos/sample_demo_1.mp4", True, "Why is this video funny"],
        #         ["example_videos/sample_demo_2.mp4", False, "Generate a creative advertisement for this product."],
        #         ["example_videos/sample_demo_3.mp4", False, "Write a poem inspired by this video."],
        #     ],
        #     css=custom_css,  # Apply custom CSS
        #     allow_flagging='auto'
        # )
        with gr.Row():
            with gr.Column():
                video_player_local = gr.Video(sources=["upload"])
                question_local = gr.Textbox(label="Your Question", placeholder="Default: What's this video talking about?")
                # has_subtitles_local = gr.Checkbox(label="Use subtitles", value=True)
                process_button_local = gr.Button("Answer the Question (QA)")
                
            with gr.Column():
                answer_local=gr.Text("Answer will be here",label="MiniGPT4-video Answer")
        try:
            process_button_local.click(fn=run_demo_local_video, inputs=[video_player_local, question_local], outputs=[answer_local])
        except Exception as ex:
            print(ex)
    with gr.Tab("Youtube videos"):
        # youtube_interface=gr.Interface(
        #     fn=gradio_demo_youtube,
        #     inputs=[gr.Textbox(label="Enter the youtube link"),gr.Checkbox(label='Use subtitles'),gr.Textbox(label="Write any Question")],
        #     outputs=["text",
        #             ],
        #     # title="<h2>YouTube videos</h2>",
        #     description="Videos length should be from one to two minutes",
        #     examples=[
        #         ["https://www.youtube.com/watch?v=8kyg5u6o21k", True, "What happens in this video?"],
        #         ["https://www.youtube.com/watch?v=zWfX5jeF6k4", True, "what is the main idea in this video?"],
        #         ["https://www.youtube.com/watch?v=W5PRZuaQ3VM", True, "Inspired by this video content suggest a creative advertisement about the same content."],
        #         ["https://www.youtube.com/watch?v=W8jcenQDXYg", True, "Describe what happens in this video."],
        #         ["https://www.youtube.com/watch?v=u3ybWiEUaUU", True, "what is creative in this video ?"],
        #         ["https://www.youtube.com/watch?v=nEwfSZfz7pw", True, "What Monica did in this video ?"],
        #     ],
        #     css=custom_css,  # Apply custom CSS
        #     allow_flagging='auto',
        # )
        with gr.Row():
            with gr.Column():
                youtube_link = gr.Textbox(label="Enter the youtube link", placeholder="Paste YouTube URL with this format 'https://www.youtube.com/watch?v=video_id'")
                video_player = gr.Video(autoplay=False)
                download_finish = gr.State(value=False)
                youtube_link.change(
                    fn=download_video,
                    inputs=[youtube_link, download_finish], 
                    outputs=[video_player, download_finish]
                )
                question = gr.Textbox(label="Your Question", placeholder="Default: What's this video talking about?")
                # has_subtitles = gr.Checkbox(label="Use subtitles", value=True)
                process_button = gr.Button("Answer the Question (QA)")
                
            with gr.Column():
                answer=gr.Text("Answer will be here",label="MiniGPT4-video Answer")
        try:
            process_button.click(fn=run_demo_youtube_video, inputs=[youtube_link, question], outputs=[answer])
        except Exception as ex:
            print(ex)
        ## Add examples to make the demo more interactive and user-friendly
        # with gr.Row():
        #     url_1=gr.Text("https://www.youtube.com/watch?v=8kyg5u6o21k")
        #     has_sub_1=True
        #     q_1=gr.Text("What happens in this video?")
        #     # add button to change the youtube link and the question with the example values
        #     use_example_1_btn=gr.Button("Use this example")
        #     use_example_1_btn.click(use_example,inputs=[url_1,has_sub_1,q_1])
            
        


if __name__ == "__main__":
    demo.queue().launch(share=True,show_error=True, server_port=2411)
