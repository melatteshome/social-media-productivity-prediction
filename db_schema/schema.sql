--
-- PostgreSQL database dump
--

-- Dumped from database version 17.4 (Debian 17.4-1.pgdg120+2)
-- Dumped by pg_dump version 17.5 (Ubuntu 17.5-1.pgdg24.04+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: social_media_users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.social_media_users (
    age integer,
    gender character varying(10),
    job_type character varying(50),
    daily_social_media_time numeric(10,4),
    social_platform_preference character varying(20),
    number_of_notifications integer,
    work_hours_per_day numeric(10,4),
    perceived_productivity_score numeric(10,4),
    actual_productivity_score numeric(10,4),
    stress_level numeric(3,1),
    sleep_hours numeric(10,4),
    screen_time_before_sleep numeric(10,4),
    breaks_during_work integer,
    uses_focus_apps boolean,
    has_digital_wellbeing_enabled boolean,
    coffee_consumption_per_day integer,
    days_feeling_burnout_per_month integer,
    weekly_offline_hours numeric(10,4),
    job_satisfaction_score numeric(10,4)
);


ALTER TABLE public.social_media_users OWNER TO postgres;

--
-- PostgreSQL database dump complete
--

